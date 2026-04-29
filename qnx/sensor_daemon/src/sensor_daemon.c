/*
 * sensor_daemon.c — BME688 (I2C) + NEO-M8N GPS (UART) publisher.
 * Publishes the latest snapshot via a QNX named channel.
 * If a sensor is absent or unreadable, the corresponding valid-flag
 * is cleared and downstream consumers know the field is unreliable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <termios.h>
#include <stdint.h>
#include <sys/neutrino.h>
#include <sys/dispatch.h>
#include <hw/i2c.h>

#include "gestsense.h"

#define BME_REG_CHIP_ID     0xD0
#define BME_REG_CALIB_1     0x8A
#define BME_REG_CALIB_2     0xE1
#define BME_REG_CTRL_HUM    0x72
#define BME_REG_CTRL_MEAS   0x74
#define BME_REG_STATUS      0x1D
#define BME_REG_DATA        0x1F
#define BME_REG_GAS         0x2A
#define BME_ID_BME688       0x61
#define BME_ID_BME680       0x60

#define SAMPLE_PERIOD_SEC   3

/* Latest snapshot. Starts with everything invalid until real data arrives. */
static sensor_reply_t g_snap = { .gps_synth = 0, .bme_synth = 0 };  /* FIX: was gps_valid / bme_valid */
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static uint8_t g_bme_addr = 0;

static void set_prio(int prio, int policy)
{
    struct sched_param p = { .sched_priority = prio };
    pthread_setschedparam(pthread_self(), policy, &p);
}

/*BME688 I2C */

static int bme_wr(int fd, uint8_t addr, uint8_t reg, uint8_t val)
{
    struct { i2c_send_t h; uint8_t d[2]; } p;
    memset(&p, 0, sizeof(p));
    p.h.slave.addr = addr; p.h.slave.fmt = I2C_ADDRFMT_7BIT;
    p.h.len = 2; p.h.stop = 1;
    p.d[0] = reg; p.d[1] = val;
    return devctl(fd, DCMD_I2C_SEND, &p, sizeof(p), NULL);
}

static int bme_rd(int fd, uint8_t addr, uint8_t reg, uint8_t *out, int n)
{
    struct { i2c_send_t h; uint8_t d[1]; } wr;
    memset(&wr, 0, sizeof(wr));
    wr.h.slave.addr = addr; wr.h.slave.fmt = I2C_ADDRFMT_7BIT;
    wr.h.len = 1; wr.h.stop = 0; wr.d[0] = reg;
    if (devctl(fd, DCMD_I2C_SEND, &wr, sizeof(wr), NULL) != EOK) return -1;

    struct { i2c_recv_t h; uint8_t d[40]; } rd;
    memset(&rd, 0, sizeof(rd));
    rd.h.slave.addr = addr; rd.h.slave.fmt = I2C_ADDRFMT_7BIT;
    rd.h.len = (uint16_t)n; rd.h.stop = 1;
    if (devctl(fd, DCMD_I2C_RECV, &rd, sizeof(rd), NULL) != EOK) return -1;
    memcpy(out, rd.d, n);
    return 0;
}

static int bme_detect(void)
{
    const char *paths[] = { "/dev/i2c1", "/dev/i2c0", "/dev/iic1", "/dev/iic0", NULL };
    const uint8_t addrs[] = { 0x77, 0x76, 0 };

    for (int i = 0; paths[i]; i++) {
        int fd = open(paths[i], O_RDWR);
        if (fd < 0) continue;
        for (int a = 0; addrs[a]; a++) {
            uint8_t id = 0;
            if (bme_rd(fd, addrs[a], BME_REG_CHIP_ID, &id, 1) != 0) continue;
            if (id == BME_ID_BME688 || id == BME_ID_BME680) {
                printf("[SENSOR] BME688 on %s addr=0x%02X\n", paths[i], addrs[a]);
                g_bme_addr = addrs[a];
                return fd;
            }
        }
        close(fd);
    }
    return -1;
}

/* Bosch compensation (BST-BME688-DS000 §3.3) */
static int32_t t_fine = 0;

static float comp_temp(int32_t adc, uint16_t T1, int16_t T2, int16_t T3)
{
    int32_t v1 = ((((adc >> 3) - ((int32_t)T1 << 1))) * (int32_t)T2) >> 11;
    int32_t v2 = (((((adc >> 4) - (int32_t)T1) * ((adc >> 4) - (int32_t)T1)) >> 12)
                  * (int32_t)T3) >> 14;
    t_fine = v1 + v2;
    return (float)((t_fine * 5 + 128) >> 8) / 100.0f;
}

static float comp_hum(int32_t adc, uint8_t H1, int16_t H2, int8_t H3,
                      int8_t H4, int8_t H5, uint8_t H6, int8_t H7)
{
    (void)H7;
    int32_t v = t_fine - 76800;
    v = (((((adc << 14) - ((int32_t)H4 << 20) - ((int32_t)H5 * v)) + 16384) >> 15)
         * (((((((v * (int32_t)H6) >> 10) * (((v * (int32_t)H3) >> 11) + 32768)) >> 10)
              + 2097152) * (int32_t)H2 + 8192) >> 14));
    v -= (((((v >> 15) * (v >> 15)) >> 7) * (int32_t)H1) >> 4);
    if (v < 0) v = 0;
    if (v > 419430400) v = 419430400;
    return (float)(v >> 12) / 1024.0f;
}

static float comp_gas(int32_t adc, uint8_t rng)
{
    static const float lut[16] = {
        1.0f, 1.0f, 1.0f,    1.0f,    1.0f, 0.99f, 1.0f,  0.992f,
        1.0f, 1.0f, 0.998f,  0.995f,  1.0f, 0.99f, 1.0f,  1.0f
    };
    float v1 = (float)((1340 + (5 * 65536)) / (float)(1340 + adc));
    return (v1 * (float)(1 << rng) * lut[rng] - 512.0f) / 1000.0f;
}

static void *bme_thread(void *arg)
{
    (void)arg;
    set_prio(PRIO_SENSOR, SCHED_RR);

    int fd = bme_detect();
    if (fd < 0) {
        printf("[SENSOR] BME688 not detected — environmental data unavailable\n");
        /* Leave bme_synth=0; retry every 30s in case it comes back */
        while (1) {
            sleep(30);
            fd = bme_detect();
            if (fd >= 0) break;
        }
    }

    usleep(20000);
    uint8_t addr = g_bme_addr;
    uint8_t c1[24] = {0}, c2[16] = {0};
    bme_rd(fd, addr, BME_REG_CALIB_1, c1, 24); usleep(5000);
    bme_rd(fd, addr, BME_REG_CALIB_2, c2, 16); usleep(5000);

    uint16_t T1 = (uint16_t)(c2[9] | ((uint16_t)c2[10] << 8));
    int16_t  T2 = (int16_t)(c1[0] | ((uint16_t)c1[1] << 8));
    int16_t  T3 = (int16_t)c1[2];
    uint8_t  H1 = (uint8_t)(((uint16_t)c2[2] << 4) | (c2[1] & 0x0F));
    int16_t  H2 = (int16_t)(((uint16_t)c2[0] << 4) | (c2[1] >> 4));
    int8_t   H3 = (int8_t)c2[3];
    int8_t   H4 = (int8_t)c2[4];
    int8_t   H5 = (int8_t)c2[5];
    uint8_t  H6 = c2[6];
    int8_t   H7 = (int8_t)c2[7];

    bme_wr(fd, addr, BME_REG_CTRL_HUM,  0x01);
    bme_wr(fd, addr, BME_REG_CTRL_MEAS, 0xB4);

    while (1) {
        bme_wr(fd, addr, BME_REG_CTRL_MEAS, 0xB5);
        uint8_t status = 0x20; int tries = 0;
        while ((status & 0x20) && tries++ < 30) {
            usleep(15000);
            bme_rd(fd, addr, BME_REG_STATUS, &status, 1);
        }
        uint8_t raw[8];
        if (bme_rd(fd, addr, BME_REG_DATA, raw, 8) != 0) {
            sleep(SAMPLE_PERIOD_SEC);
            continue;
        }

        int32_t adc_T = ((int32_t)raw[3] << 12) | ((int32_t)raw[4] << 4) | ((int32_t)raw[5] >> 4);
        int32_t adc_H = ((int32_t)raw[6] << 8) | (int32_t)raw[7];
        float tc  = comp_temp(adc_T, T1, T2, T3);
        float hum = comp_hum(adc_H, H1, H2, H3, H4, H5, H6, H7);

        float gas = 0.0f;
        uint8_t gr[2];
        int gas_ok = 0;
        if (bme_rd(fd, addr, BME_REG_GAS, gr, 2) == 0) {
            int32_t adc_G = ((int32_t)gr[0] << 2) | ((int32_t)(gr[1] & 0xC0) >> 6);
            gas = comp_gas(adc_G, gr[1] & 0x0F);
            gas_ok = 1;
        }

        pthread_mutex_lock(&g_mtx);
        g_snap.temp = tc;
        g_snap.hum  = hum;
        g_snap.gas  = gas;
        g_snap.bme_synth = gas_ok;   /* FIX: was bme_valid */
        pthread_mutex_unlock(&g_mtx);

        printf("[SENSOR] BME688 %.1f°C %.1f%%RH %.1fkΩ\n", tc, hum, gas);
        fflush(stdout);
        sleep(SAMPLE_PERIOD_SEC);
    }
    return NULL;
}

/*NEO-M8N NMEA parser */

static double nmea_to_deg(const char *raw, char hem)
{
    if (!raw || strlen(raw) < 4) return 0.0;
    double v = atof(raw);
    int deg = (int)(v / 100);
    double r = deg + (v - deg * 100.0) / 60.0;
    return (hem == 'S' || hem == 'W') ? -r : r;
}

static void gnss_parse(const char *line)
{
    char buf[256];
    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char *star = strchr(buf, '*');
    if (star) *star = 0;

    char *f[15]; int n = 0;
    char *tok = strtok(buf, ",");
    while (tok && n < 15) { f[n++] = tok; tok = strtok(NULL, ","); }
    if (n < 10) return;
    if (atoi(f[6]) == 0) return;   /* fix quality 0 = no fix */

    double lat = nmea_to_deg(f[2], f[3][0]);
    double lon = nmea_to_deg(f[4], f[5][0]);
    double alt = strlen(f[9]) ? atof(f[9]) : 0.0;
    if (lat == 0.0 && lon == 0.0) return;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return;

    pthread_mutex_lock(&g_mtx);
    g_snap.lat = lat;
    g_snap.lon = lon;
    g_snap.alt = alt;
    g_snap.sats = atoi(f[7]);
    g_snap.gps_synth = 1;           /* FIX: was gps_valid */
    pthread_mutex_unlock(&g_mtx);

    printf("[SENSOR] GPS fix %.6f,%.6f alt=%.1fm sats=%d\n",
           lat, lon, alt, atoi(f[7]));
    fflush(stdout);
}

static void *gnss_thread(void *arg)
{
    (void)arg;
    set_prio(PRIO_SENSOR, SCHED_RR);

    int fd = open("/dev/ser1", O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        printf("[SENSOR] /dev/ser1 unavailable — GPS data will remain invalid\n");
        return NULL;
    }

    struct termios t;
    tcgetattr(fd, &t);
    cfsetispeed(&t, B9600); cfsetospeed(&t, B9600);
    t.c_cflag = CS8 | CLOCAL | CREAD;
    t.c_iflag = t.c_oflag = t.c_lflag = 0;
    t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 10;
    tcsetattr(fd, TCSANOW, &t);
    fcntl(fd, F_SETFL, 0);

    printf("[SENSOR] GPS listener ready on /dev/ser1\n");
    fflush(stdout);

    char line[256]; int li = 0; char c;
    while (1) {
        if (read(fd, &c, 1) <= 0) { usleep(5000); continue; }
        if (c == '\n') {
            line[li] = 0;
            if (!strncmp(line, "$GPGGA", 6) || !strncmp(line, "$GNGGA", 6))
                gnss_parse(line);
            li = 0;
        } else if (c != '\r' && li < (int)sizeof(line) - 1) {
            line[li++] = c;
        }
    }
    close(fd);
    return NULL;
}

/* main  */

int main(void)
{
    printf("[SENSOR] sensor_daemon starting\n");
    fflush(stdout);

    pthread_t t_bme, t_gnss;
    pthread_create(&t_bme,  NULL, bme_thread,  NULL);
    pthread_create(&t_gnss, NULL, gnss_thread, NULL);

    name_attach_t *att = name_attach(NULL, SVC_SENSOR, 0);
    if (!att) { perror("[SENSOR] name_attach"); return 1; }
    printf("[SENSOR] registered service '%s' chid=%d\n", SVC_SENSOR, att->chid);
    fflush(stdout);

    sensor_req_t req;
    sensor_reply_t rep;
    while (1) {
        int rcvid = MsgReceive(att->chid, &req, sizeof(req), NULL);
        if (rcvid == 0) continue;
        if (rcvid == -1) { perror("[SENSOR] MsgReceive"); break; }

        if (req.type == MSG_SENSOR_GET) {
            pthread_mutex_lock(&g_mtx);
            rep = g_snap;
            pthread_mutex_unlock(&g_mtx);
            MsgReply(rcvid, 0, &rep, sizeof(rep));
        } else {
            MsgError(rcvid, ENOSYS);
        }
    }

    name_detach(att, 0);
    return 0;
}
