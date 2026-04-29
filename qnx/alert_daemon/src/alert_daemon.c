/*
 * alert_daemon.c — emergency dispatch orchestrator.
 * Receives gesture alerts over UDP, queries sensor_daemon,
 * dispatches io_daemon, applies cascade routing, forwards to
 * dashboard. Cancel paths (AI fist gesture + dashboard ACK)
 * converge on the same PULSE_ACK to io_daemon.
 * * UPDATED: Dynamic Dashboard IP discovery and ACK-thread learning.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/neutrino.h>
#include <sys/dispatch.h>
#include <sys/syspage.h>

#define ENABLE_APS              1
#define APS_BUDGET_PERCENT      40
#define APS_CRITICAL_BUDGET_MS  100

/* APS partition types and SchedCtl constants were removed entirely in
 * QNX 8.0 (_NTO_VERSION >= 800).  Only include the old header and
 * compile the APS code-path when building against QNX 7.x. */
#if ENABLE_APS && defined(_NTO_VERSION) && (_NTO_VERSION < 800)
  #if defined(__has_include) && __has_include(<sys/aps.h>)
    #include <sys/aps.h>
  #endif
  #define _APS_SUPPORTED 1
#else
  #define _APS_SUPPORTED 0
#endif

#include "gestsense.h"

static int   g_sensor_coid = -1;
static int   g_io_coid     = -1;
static int   g_udp_fd      = -1;
static int   g_ack_fd      = -1;

/* UPDATED: Locked set to 0 to allow dynamic IP discovery from server.py */
static char  g_dashboard_ip[64] = DASHBOARD_HOST;
static int   g_dashboard_locked = 0;

static uint64_t g_total_alerts   = 0;
static uint64_t g_total_cancels  = 0;
static uint64_t g_total_cascades = 0;
static uint64_t g_latency_sum_ns = 0;
static uint64_t g_latency_count  = 0;
static uint64_t g_latency_max_ns = 0;
static pthread_mutex_t g_metrics_mtx = PTHREAD_MUTEX_INITIALIZER;

/* Cascade rules: FIRE and DISTRESS auto-dispatch medical + police */
typedef struct {
    const char *primary;
    const char *cascade[4];
    const char *cascade_msg[4];
} cascade_rule_t;

static const cascade_rule_t CASCADE_TABLE[] = {
    { .primary = "FIRE",
      .cascade = { "AMBULANCE", "POLICE", NULL },
      .cascade_msg = { "!! MEDICAL STANDBY (fire call) !!",
                       "!! CROWD CONTROL (fire call) !!", NULL } },
    { .primary = "DISTRESS",
      .cascade = { "AMBULANCE", "POLICE", NULL },
      .cascade_msg = { "!! MEDICAL STANDBY (distress call) !!",
                       "!! SEARCH & RESCUE (distress call) !!", NULL } },
    { .primary = NULL }
};

static void set_prio(int prio, int policy)
{
    struct sched_param p = { .sched_priority = prio };
    pthread_setschedparam(pthread_self(), policy, &p);
}

/* Adaptive Partitioning (QNX-exclusive) */
static void setup_adaptive_partition(void)
{
#if _APS_SUPPORTED
    sched_aps_create_parms parms;
    memset(&parms, 0, sizeof(parms));
    strncpy(parms.name, "alerts", _NTO_PARTITION_NAME_LENGTH - 1);
    parms.budget_percent     = APS_BUDGET_PERCENT;
    parms.critical_budget_ms = APS_CRITICAL_BUDGET_MS;

    int pid = SchedCtl(SCHED_APS_CREATE_PARTITION, &parms, sizeof(parms));
    if (pid == -1) {
        printf("[APS] kernel lacks APS support (errno=%d)\n", errno);
        return;
    }
    sched_aps_join_parms join = { .id = pid, .pid = 0 };
    if (SchedCtl(SCHED_APS_JOIN_PARTITION, &join, sizeof(join)) == -1)
        printf("[APS] join failed (errno=%d)\n", errno);
    else
        printf("[APS] joined 'alerts' id=%d budget=%d%%\n",
               pid, parms.budget_percent);
#else
    printf("[APS] skipped — not supported on QNX 8.0+\n");
#endif
}

/* minimal JSON helpers */
static int json_get_str(const char *json, const char *key, char *out, int outlen)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return -1;
    p = strchr(p + strlen(pat), ':');
    if (!p) return -1;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < outlen - 1) out[i++] = *p++;
    out[i] = 0;
    return 0;
}

static double json_get_num(const char *json, const char *key, double fb)
{
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return fb;
    p = strchr(p + strlen(pat), ':');
    if (!p) return fb;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atof(p);
}

/* IPC helpers */
static int open_service(const char *name)
{
    int coid = -1, retries = 0;
    while (coid < 0) {
        coid = name_open(name, 0);
        if (coid >= 0) return coid;
        if (retries == 0) printf("[ALERT] waiting for %s\n", name);
        fflush(stdout);
        sleep(1);
        if (++retries > 60) return -1;
    }
    return coid;
}

static int query_sensor(sensor_reply_t *out)
{
    sensor_req_t req = { .type = MSG_SENSOR_GET };
    return MsgSend(g_sensor_coid, &req, sizeof(req), out, sizeof(*out));
}

static int dispatch_io(const char *gesture, const char *person,
                       const char *msg, const sensor_reply_t *snap)
{
    io_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type = MSG_IO_ALERT;
    strncpy(cmd.gesture,   gesture, sizeof(cmd.gesture)   - 1);
    strncpy(cmd.person,    person,  sizeof(cmd.person)    - 1);
    strncpy(cmd.alert_msg, msg,     sizeof(cmd.alert_msg) - 1);
    cmd.lat       = snap->lat;
    cmd.lon       = snap->lon;
    cmd.gps_synth = snap->gps_synth;
    cmd.temp      = snap->temp;
    cmd.hum       = snap->hum;

    io_reply_t rep;
    return MsgSend(g_io_coid, &cmd, sizeof(cmd), &rep, sizeof(rep));
}

static int send_ack_pulse(void)
{
    return MsgSendPulse(g_io_coid, PRIO_IO_MSG, PULSE_ACK, 0);
}

static void send_udp(const char *json, int len)
{
    struct sockaddr_in dst;
    memset(&dst, 0, sizeof(dst));
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(DASHBOARD_PORT);
    inet_pton(AF_INET, g_dashboard_ip, &dst.sin_addr);
    sendto(g_udp_fd, json, len, 0, (struct sockaddr *)&dst, sizeof(dst));
}

/* JSON builders */
static int build_alert_json(char *buf, int buflen,
                            const char *gesture, const char *person,
                            const char *msg, int cascade,
                            const char *cascade_of,
                            const sensor_reply_t *snap, double ts)
{
    return snprintf(buf, buflen,
        "{\"type\":\"ALERT\","
        "\"gesture\":\"%s\",\"person\":\"%s\",\"alert\":\"%s\","
        "\"timestamp\":%.3f,"
        "\"cascade\":%s,\"cascade_of\":\"%s\","
        "\"lat\":%.6f,\"lon\":%.6f,\"alt\":%.1f,\"gps_synth\":%d,"
        "\"temp\":%.1f,\"hum\":%.1f,\"gas\":%.1f,\"bme_synth\":%d}",
        gesture, person, msg, ts,
        cascade ? "true" : "false",
        cascade ? cascade_of : "",
        snap->lat, snap->lon, snap->alt, snap->gps_synth,
        snap->temp, snap->hum, snap->gas, snap->bme_synth);
}

static int build_cancel_json(char *buf, int buflen,
                             const char *gesture, const char *person, double ts)
{
    return snprintf(buf, buflen,
        "{\"type\":\"CANCEL\",\"gesture\":\"%s\",\"person\":\"%s\","
        "\"timestamp\":%.3f,\"reason\":\"operator_cancel\"}",
        gesture, person, ts);
}

/* alert pipeline */
static void handle_alert(const char *json, const struct sockaddr_in *from)
{
    uint64_t t_start = ClockCycles();

    /* Learning dashboard IP from source of incoming alert (optional but helpful) */
    if (!g_dashboard_locked) {
        inet_ntop(AF_INET, &from->sin_addr, g_dashboard_ip, sizeof(g_dashboard_ip));
        g_dashboard_locked = 1;
        printf("[ALERT] dashboard locked to %s:%d\n", g_dashboard_ip, DASHBOARD_PORT);
    }

    char gesture[32] = "", person[16] = "", msg[96] = "";
    if (json_get_str(json, "gesture", gesture, sizeof(gesture)) < 0) return;
    if (json_get_str(json, "person",  person,  sizeof(person))  < 0) return;
    if (json_get_str(json, "alert",   msg,     sizeof(msg))     < 0)
        strncpy(msg, "EMERGENCY", sizeof(msg) - 1);
    double ts = json_get_num(json, "timestamp", (double)time(NULL));

    printf("[ALERT] %s %s \"%s\"\n", gesture, person, msg);

    sensor_reply_t snap;
    memset(&snap, 0, sizeof(snap));
    if (query_sensor(&snap) == -1) {
        printf("[ALERT] sensor query failed — flags cleared\n");
        snap.gps_synth = 0;
        snap.bme_synth = 0;
    }

    char out[1024];
    int n = build_alert_json(out, sizeof(out), gesture, person, msg, 0, "", &snap, ts);
    send_udp(out, n);

    int cascade_count = 0;
    for (int i = 0; CASCADE_TABLE[i].primary; i++) {
        if (strcmp(CASCADE_TABLE[i].primary, gesture) != 0) continue;
        for (int k = 0; CASCADE_TABLE[i].cascade[k]; k++) {
            n = build_alert_json(out, sizeof(out), CASCADE_TABLE[i].cascade[k], person,
                                 CASCADE_TABLE[i].cascade_msg[k], 1, gesture, &snap, ts);
            send_udp(out, n);
            printf("[CASCADE] + %s\n", CASCADE_TABLE[i].cascade[k]);
            cascade_count++;
        }
        break;
    }

    if (dispatch_io(gesture, person, msg, &snap) == -1)
        printf("[ALERT] io_daemon dispatch failed\n");

    uint64_t cycles = ClockCycles() - t_start;
    uint64_t cps    = SYSPAGE_ENTRY(qtime)->cycles_per_sec;
    uint64_t ns     = (cps > 0) ? (cycles * 1000000000ULL) / cps : 0;

    pthread_mutex_lock(&g_metrics_mtx);
    g_total_alerts++;
    g_total_cascades += cascade_count;
    g_latency_sum_ns += ns;
    g_latency_count++;
    if (ns > g_latency_max_ns) g_latency_max_ns = ns;
    pthread_mutex_unlock(&g_metrics_mtx);

    printf("[ALERT] dispatched in %lu ns (cascade=%d)\n", (unsigned long)ns, cascade_count);
    fflush(stdout);
}

static void handle_cancel(const char *json)
{
    char gesture[32] = "", person[16] = "";
    json_get_str(json, "gesture", gesture, sizeof(gesture));
    json_get_str(json, "person",  person,  sizeof(person));
    double ts = json_get_num(json, "timestamp", (double)time(NULL));

    printf("[CANCEL] %s %s\n", gesture, person);

    if (send_ack_pulse() == -1)
        printf("[CANCEL] PULSE_ACK failed\n");

    char out[512];
    int n = build_cancel_json(out, sizeof(out), gesture, person, ts);
    send_udp(out, n);

    pthread_mutex_lock(&g_metrics_mtx);
    g_total_cancels++;
    pthread_mutex_unlock(&g_metrics_mtx);
    fflush(stdout);
}

/* threads */
static void *ack_thread(void *arg)
{
    (void)arg;
    set_prio(PRIO_ACK, SCHED_FIFO);
    printf("[ALERT] ack_thread prio=%d\n", PRIO_ACK);

    char buf[2048];
    while (1) {
        struct sockaddr_in from;
        socklen_t sl = sizeof(from);
        /* UPDATED: recvfrom to capture PC IP address */
        int n = recvfrom(g_ack_fd, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &sl);
        if (n <= 0) { usleep(1000); continue; }
        buf[n] = 0;

        /* UPDATED: Learn the dashboard IP from the PC that clicks ACK */
        if (!g_dashboard_locked) {
            inet_ntop(AF_INET, &from.sin_addr, g_dashboard_ip, sizeof(g_dashboard_ip));
            g_dashboard_locked = 1;
            printf("[ACK] learned dashboard IP: %s\n", g_dashboard_ip);
        }

        char gesture[32] = "", person[16] = "";
        json_get_str(buf, "gesture", gesture, sizeof(gesture));
        json_get_str(buf, "person",  person,  sizeof(person));

        printf("[ACK] dashboard ACK %s\n", gesture);

        if (send_ack_pulse() == -1) printf("[ACK] PULSE_ACK failed\n");

        char out[512];
        int len = build_cancel_json(out, sizeof(out), gesture[0] ? gesture : "UNKNOWN", person, (double)time(NULL));
        send_udp(out, len);

        pthread_mutex_lock(&g_metrics_mtx);
        g_total_cancels++;
        pthread_mutex_unlock(&g_metrics_mtx);
        fflush(stdout);
    }
    return NULL;
}

static void *udp_thread(void *arg)
{
    (void)arg;
    set_prio(PRIO_UDP_RECV, SCHED_FIFO);
    printf("[ALERT] udp_thread prio=%d\n", PRIO_UDP_RECV);

    char buf[2048];
    while (1) {
        struct sockaddr_in src;
        socklen_t sl = sizeof(src);
        int n = recvfrom(g_udp_fd, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&src, &sl);
        if (n <= 0) { usleep(1000); continue; }
        buf[n] = 0;

        char type[16] = "";
        json_get_str(buf, "type", type, sizeof(type));

        if (type[0] == 0 || strcmp(type, "ALERT") == 0)
            handle_alert(buf, &src);
        else if (strcmp(type, "CANCEL") == 0)
            handle_cancel(buf);
    }
    return NULL;
}

static void *metrics_thread(void *arg)
{
    (void)arg;
    set_prio(PRIO_STATUS, SCHED_OTHER);
    while (1) {
        sleep(10);
        pthread_mutex_lock(&g_metrics_mtx);
        uint64_t a = g_total_alerts, c = g_total_cancels,
                 csc = g_total_cascades,
                 avg = g_latency_count ? g_latency_sum_ns / g_latency_count : 0,
                 mx = g_latency_max_ns;
        pthread_mutex_unlock(&g_metrics_mtx);

        if (a > 0 || c > 0)
            printf("[METRICS] alerts=%lu cancels=%lu cascades=%lu avg=%lu ns max=%lu ns\n",
                   (unsigned long)a, (unsigned long)c, (unsigned long)csc, (unsigned long)avg, (unsigned long)mx);
        fflush(stdout);
    }
    return NULL;
}

/* main */
static int bind_udp(int port)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int main(void)
{
    printf("[ALERT] alert_daemon starting\n");
    fflush(stdout);

    setup_adaptive_partition();

    g_sensor_coid = open_service(SVC_SENSOR);
    g_io_coid     = open_service(SVC_IO);
    if (g_sensor_coid < 0 || g_io_coid < 0) {
        fprintf(stderr, "[ALERT] service connect failed\n");
        return 1;
    }
    printf("[ALERT] %s coid=%d  %s coid=%d\n", SVC_SENSOR, g_sensor_coid, SVC_IO, g_io_coid);

    g_udp_fd = bind_udp(UDP_ALERT_PORT);
    g_ack_fd = bind_udp(ACK_PORT);
    if (g_udp_fd < 0 || g_ack_fd < 0) {
        perror("[ALERT] bind");
        return 1;
    }
    printf("[ALERT] UDP :%d (alerts)  :%d (ACK)\n", UDP_ALERT_PORT, ACK_PORT);
    fflush(stdout);

    pthread_t t_udp, t_ack, t_metrics;
    pthread_create(&t_udp,     NULL, udp_thread,     NULL);
    pthread_create(&t_ack,     NULL, ack_thread,     NULL);
    pthread_create(&t_metrics, NULL, metrics_thread, NULL);

    pthread_join(t_udp, NULL);

    close(g_udp_fd);
    close(g_ack_fd);
    name_close(g_sensor_coid);
    name_close(g_io_coid);
    return 0;
}
