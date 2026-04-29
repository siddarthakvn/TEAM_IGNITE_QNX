/*
 * gestsense.h  ─  Shared definitions for the 3-daemon GestSense system
 * Team Ignite | Q-eHACK 2026
 *
 * Architecture:
 *   ┌──────────────┐   MsgSend     ┌──────────────┐   MsgSend    ┌─────────────┐
 *   │  alert_d     │ ─────────────▶│  sensor_d    │              │   io_d      │
 *   │  (priority 25)│◀──MsgReply───│  (priority 15)│              │  (priority 22)│
 *   │              │                └──────────────┘              │             │
 *   │              │ ───────────── MsgSend ─────────────────────▶│             │
 *   │              │◀──────────────── MsgReply ──────────────────│             │
 *   │              │                                              │             │
 *   │              │ ──────────── MsgSendPulse (ACK) ───────────▶│             │
 *   └──────────────┘                                              └─────────────┘
 *          ▲                                                             │
 *          │ UDP                                                        │ GPIO + GSM
 *      app.py (Win)                                                 LED/Buzzer/SMS
 *
 * Why split into 3 processes?
 *   1. Fault isolation — io_daemon crash does not kill alert_daemon
 *   2. Priority separation — RTOS scheduler can preempt properly
 *   3. Proper QNX IPC — demonstrates name_attach/MsgSend/Pulses
 */

#ifndef GESTSENSE_H
#define GESTSENSE_H

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════
 *  SERVICE NAMES (for name_attach / name_open)
 * ═══════════════════════════════════════════════════════════ */
#define SVC_SENSOR   "gestsense/sensor"
#define SVC_IO       "gestsense/io"

/* ═══════════════════════════════════════════════════════════
 *  NETWORK CONFIG
 * ═══════════════════════════════════════════════════════════ */
#define UDP_ALERT_PORT   5005    /* app.py → alert_daemon         */
#define DASHBOARD_PORT   5006    /* alert_daemon → server.py      */
#define ACK_PORT         5007    /* server.py → alert_daemon      */
#define DASHBOARD_HOST   "10.0.0.1"

/* ═══════════════════════════════════════════════════════════
 *  GSM (SIM800L)
 * ═══════════════════════════════════════════════════════════ */
#define GSM_PORT         "/dev/ser2"
#define ALERT_NUMBER     "+919999999999"

/* ═══════════════════════════════════════════════════════════
 *  HARDCODED FALLBACK VALUES  (used if hardware missing)
 *  Location: 17.254973°N, 78.308165°E   (Hyderabad)
 *  Sensor:   April pre-monsoon daytime conditions
 * ═══════════════════════════════════════════════════════════ */
#define SYNTH_LAT        17.254973
#define SYNTH_LON        78.308165
#define SYNTH_ALT        512.0
#define SYNTH_SATS       8
#define SYNTH_TEMP       36.5f    /* April Hyderabad daytime (°C)    */
#define SYNTH_HUM        32.0f    /* dry pre-monsoon (%)             */
#define SYNTH_GAS        180.0f   /* clean outdoor air (kOhm)        */

/* ═══════════════════════════════════════════════════════════
 *  GPIO — BCM numbering on RPi4
 * ═══════════════════════════════════════════════════════════ */
#define GPIO_PHYS_BASE   0xFE200000UL
#define GPIO_MAP_LEN     0x00001000UL
#define PIN_RED          17
#define PIN_GREEN        27
#define PIN_BUZZER       19

/* ═══════════════════════════════════════════════════════════
 *  REAL-TIME PRIORITIES  (SCHED_FIFO)
 *
 *  Higher number = more urgent. QNX will preempt lower-priority
 *  threads instantly when a higher-priority thread becomes ready.
 * ═══════════════════════════════════════════════════════════ */
#define PRIO_UDP_RECV    25   /* Emergency ingress — absolute top   */
#define PRIO_ACK         23   /* Stop buzzer fast on ACK            */
#define PRIO_IO_PATTERN  22   /* Deterministic LED/buzzer timing    */
#define PRIO_IO_MSG      21   /* IO message receiver                */
#define PRIO_SENSOR      15   /* Sensor polling — not time-critical */
#define PRIO_GSM         10   /* Blocking serial I/O                */
#define PRIO_STATUS      5    /* Logging / metrics                  */

/* ═══════════════════════════════════════════════════════════
 *  MESSAGE TYPES — alert_daemon ↔ io_daemon
 * ═══════════════════════════════════════════════════════════ */
#define MSG_IO_ALERT     0x1001  /* start pattern + send SMS        */
#define MSG_IO_STATUS    0x1002  /* query IO state                  */

/* Pulse codes */
#define PULSE_ACK        1       /* ACK received — stop pattern     */

/* ═══════════════════════════════════════════════════════════
 *  MESSAGE TYPES — alert_daemon ↔ sensor_daemon
 * ═══════════════════════════════════════════════════════════ */
#define MSG_SENSOR_GET   0x2001  /* return latest sensor snapshot   */

/* ═══════════════════════════════════════════════════════════
 *  WIRE FORMATS
 * ═══════════════════════════════════════════════════════════ */
typedef struct {
    uint16_t type;               /* MSG_IO_ALERT                   */
    char     gesture[32];        /* AMBULANCE / POLICE / FIRE / ... */
    char     person[16];
    char     alert_msg[64];
    double   lat, lon;
    float    temp, hum;
    int      gps_synth;
    uint64_t t_enqueue_ns;       /* for RT latency measurement     */
} io_cmd_t;

typedef struct {
    int      ok;
    uint64_t t_received_ns;      /* echoed back for latency calc   */
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
