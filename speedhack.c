#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

#define CLOCK_COUNT 12
#define NS_IN_S 1000000000
static struct timespec base_times[CLOCK_COUNT] = {0};
static int base_times_init[CLOCK_COUNT] = {0};
static pthread_mutex_t mutex;

int clock_gettime(clockid_t clk_id, struct timespec *tp) {
  pthread_mutex_lock(&mutex);

  // Pointer to the original clock_gettime function
  static int (*original_clock_gettime)(clockid_t clk_id, struct timespec *tp) =
      NULL;

  // Get a pointer to the original function on the first call
  if (original_clock_gettime == NULL) {
    original_clock_gettime = (int (*)(clockid_t, struct timespec *))dlsym(
        RTLD_NEXT, "clock_gettime");
  }

  int ret = original_clock_gettime(clk_id, tp);

  assert(clk_id < CLOCK_COUNT);
  if (!base_times_init[clk_id]) {
    base_times[clk_id] = *tp;
    base_times_init[clk_id] = 1;
  }

  pthread_mutex_unlock(&mutex);

  __int128_t s = tp->tv_sec - base_times[clk_id].tv_sec;
  __int128_t ns = tp->tv_nsec - base_times[clk_id].tv_nsec;
  __int128_t total_ns = s * NS_IN_S + ns;
  total_ns /= 2;

  tp->tv_sec = total_ns / NS_IN_S + base_times[clk_id].tv_sec;
  tp->tv_nsec = total_ns % NS_IN_S + base_times[clk_id].tv_nsec;

  return ret;
}
