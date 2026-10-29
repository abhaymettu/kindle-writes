
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdint.h>
#define W 1072
#define H 1448
struct mxcfb_rect { uint32_t top, left, width, height; };
struct mxcfb_alt_buffer_data { void *virt_addr; uint32_t width, height; struct mxcfb_rect alt_update_region; };
struct mxcfb_swipe_data { uint32_t direction; uint32_t steps; };
struct mxcfb_update_data_mtk {
    struct mxcfb_rect update_region;
    uint32_t waveform_mode;
    uint32_t update_mode;
    uint32_t update_marker;
    int temp;
    unsigned flags;
    int dither_mode;
    int quant_bit;
    struct mxcfb_alt_buffer_data alt_buffer_data;
    struct mxcfb_swipe_data swipe_data;
    uint32_t hist_bw_waveform_mode;
    uint32_t hist_gray_waveform_mode;
    uint32_t ts_pxp;
    uint32_t ts_epdc;
};
struct mxcfb_update_marker_data { uint32_t update_marker; uint32_t collision_test; };
#define MXCFB_SEND_UPDATE_MTK 1080051246u
#define MXCFB_WAIT_FOR_UPDATE_COMPLETE 3221767727u
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/mnt/base-us/kindle-writes/screen.raw";
    int f = open(path, O_RDONLY);
    if (f < 0) { perror("open raw"); return 1; }
    unsigned char *buf = malloc((size_t)W*H);
    ssize_t n = read(f, buf, (size_t)W*H);
    close(f);
    if (n != (ssize_t)W*H) { fprintf(stderr, "raw size %zd != %d\n", n, W*H); return 1; }
    int fb = open("/dev/fb0", O_RDWR);
    if (fb < 0) { perror("open fb0"); return 1; }
    unsigned char *mem = mmap(NULL, (size_t)W*H, PROT_READ|PROT_WRITE, MAP_SHARED, fb, 0);
    if (mem == MAP_FAILED) { perror("mmap"); return 1; }
    memcpy(mem, buf, (size_t)W*H);
    struct mxcfb_update_data_mtk u;
    memset(&u, 0, sizeof u);
    u.update_region.top = 0; u.update_region.left = 0;
    u.update_region.width = W; u.update_region.height = H;
    u.waveform_mode = 2;      /* MTK_WAVEFORM_MODE_GC16 */
    u.update_mode = 1;        /* full */
    u.update_marker = 1;
    u.temp = 0x1000;          /* TEMP_USE_AMBIENT */
    int r = ioctl(fb, MXCFB_SEND_UPDATE_MTK, &u);
    if (r < 0) { perror("MXCFB_SEND_UPDATE_MTK"); munmap(mem,(size_t)W*H); close(fb); return 2; }
    struct mxcfb_update_marker_data mk; mk.update_marker = 1; mk.collision_test = 0;
    int w = ioctl(fb, MXCFB_WAIT_FOR_UPDATE_COMPLETE, &mk);
    if (w < 0) perror("WAIT_FOR_UPDATE_COMPLETE");
    printf("mtk update sent, wait rc=%d (sizeof=%zu)\n", w, sizeof u);
    munmap(mem, (size_t)W*H);
    close(fb);
    return 0;
}
