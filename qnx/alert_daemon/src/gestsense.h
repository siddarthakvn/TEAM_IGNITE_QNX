#ifndef GESTSENSE_H
#define GESTSENSE_H

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════
 * SERVICE NAMES
 * ═══════════════════════════════════════════════════════════ */
#define SVC_SENSOR   "gestsense/sensor"
#define SVC_IO       "gestsense/io"

/* ═══════════════════════════════════════════════════════════
 * NETWORK CONFIG
 * ═══════════════════════════════════════════════════════════ */
#define UDP_ALERT_PORT   5005    /* app.py → alert_daemon         */
#define DASHBOARD_PORT   5006    /* alert_daemon → server.py (UDP_LISTEN_PORT) */
#define ACK_PORT         5007    /* server.py → alert_daemon (QNX_ACK_PORT)    */

/* CHANGE THIS: Set this to the IP of your PC running server.py */
#define DASHBOARD_HOST   "192.168.x.x"

/* ═══════════════════════════════════════════════════════════
 * GSM (SIM800L)
 * ═══════════════════════════════════════════════════════════ */
#define GSM_PORT         "/dev/ser2"
#define ALERT_NUMBER     "+919999999999"

/* ═══════════════════════════════════════════════════════════
 * SYNTHETIC DATA FALLBACKS
 * ═══════════════════════════════════════════════════════════ */
#define SYNTH_LAT        17.254973
#define SYNTH_LON        78.308165
#define SYNTH_ALT        512.0
#define SYNTH_SATS       8
#define SYNTH_TEMP       36.5f
#define SYNTH_HUM        32.0f
#define SYNTH_GAS        180.0f

/* ═══════════════════════════════════════════════════════════
 * GPIO — BCM numbering on RPi4
 * ═══════════════════════════════════════════════════════════ */
#define GPIO_PHYS_BASE   0xFE200000UL
#define GPIO_MAP_LEN     0x00001000UL
#define PIN_RED          17
#define PIN_GREEN        27
#define PIN_BUZZER       19

/* ═══════════════════════════════════════════════════════════
 * REAL-TIME PRIORITIES
 * ═══════════════════════════════════════════════════════════ */
#define PRIO_UDP_RECV    25
#define PRIO_ACK         23
#define PRIO_IO_PATTERN  22
#define PRIO_IO_MSG      21
#define PRIO_SENSOR      15
#define PRIO_GSM         10
#define PRIO_STATUS      5

/* ═══════════════════════════════════════════════════════════
 * MESSAGE TYPES & WIRE FORMATS
 * ═══════════════════════════════════════════════════════════ */
#define MSG_IO_ALERT     0x1001
#define MSG_IO_STATUS    0x1002
#define PULSE_ACK        1
#define MSG_SENSOR_GET   0x2001

typedef struct {
    uint16_t type;
    char     gesture[32];
    char     person[16];
    char     alert_msg[64];
    double   lat, lon;
    float    temp, hum;
    int      gps_synth;
    uint64_t t_enqueue_ns;
} io_cmd_t;

typedef struct {
    int      ok;
    uint64_t t_received_ns;
} io_reply_t;

typedef struct { uint16_t type; } sensor_req_t;

typedef struct {
    double lat, lon, alt;
    int    sats;
    int    gps_synth;
    float  temp, hum, gas;
    int    bme_synth;
} sensor_reply_t;

#endif /* GESTSENSE_H */
