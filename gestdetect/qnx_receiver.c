/*
 * qnx_receiver.c  —  UDP gesture alert receiver for QNX (RPi4)
 *
 * Compile on QNX:
 *   qcc -o qnx_receiver qnx_receiver.c
 *   (or cross-compile: aarch64-unknown-nto-qnx7.1.0-gcc -o qnx_receiver qnx_receiver.c)
 *
 * Run:
 *   ./qnx_receiver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>

#define PORT    5005
#define BUFSIZE 1024

static void timestamp(char *buf, size_t n) {
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    strftime(buf, n, "%H:%M:%S", tm);
}

int main(void) {
    int fd;
    struct sockaddr_in addr, sender;
    char buf[BUFSIZE];
    char ts[16];
    ssize_t n;
    socklen_t slen = sizeof(sender);

    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("socket"); return 1; }

    /* Allow quick restart without "address already in use" */
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(PORT);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return 1;
    }

    printf("[QNX receiver] Listening on UDP port %d\n", PORT);
    fflush(stdout);

    for (;;) {
        n = recvfrom(fd, buf, BUFSIZE - 1, 0,
                     (struct sockaddr *)&sender, &slen);
        if (n < 0) { perror("recvfrom"); continue; }
        buf[n] = '\0';

        timestamp(ts, sizeof(ts));
        printf("[%s] ALERT from %s  =>  %s\n",
               ts, inet_ntoa(sender.sin_addr), buf);
        fflush(stdout);

        /* Echo payload back so the PC side can compute RTT. */
        ssize_t sent = sendto(fd, buf, n, 0,
                              (struct sockaddr *)&sender, slen);
        if (sent < 0) {
            perror("sendto");
        }

        /*
         * TODO: add your QNX-specific response here, e.g.:
         *   - Write to a QNX message queue
         *   - Toggle a GPIO pin
         *   - Send a QNX pulse to another process
         */
    }

    close(fd);
    return 0;
}
