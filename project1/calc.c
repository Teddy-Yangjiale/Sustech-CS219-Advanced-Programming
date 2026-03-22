#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <readline/history.h>
#include <readline/readline.h>

#define BASE 1000000000
#define DECIMAL_PLACES 50 
#define MAX_BUF_SIZE 150000

#ifndef M_PI_L
#define M_PI_L 3.141592653589793238462643383279502884L
#endif


typedef struct {
  uint32_t* dig;
  uint32_t len;
  int32_t sign;
} BigInt;

typedef struct {
  long double r, i;
} Complex;

typedef struct {
  BigInt* val;
  int scale;
} BigDec;

typedef enum {
  TOK_NUM,
  TOK_PLUS,
  TOK_MINUS,
  TOK_MUL,
  TOK_DIV,
  TOK_LPAREN,
  TOK_RPAREN,
  TOK_EOF,
  TOK_FUNC
} TokenType;

typedef struct {
  TokenType type;
  char val_str[MAX_BUF_SIZE];
  size_t pos;
} Token;

// ============================================================================
// 全局变量声明 (Global Variables)
// ============================================================================
int g_err_flag = 0;
const char* expr_start = NULL;
const char* expr_ptr;
Token curr_tok;

// ============================================================================
// 函数前向声明 (Forward Declarations)
// ============================================================================

void set_error(const char* msg);

BigInt* bi_new();
void bi_free(BigInt* a);
void bi_trim(BigInt* a);
BigInt* bi_copy(const BigInt* a);
void bi_shift_blocks_left(BigInt* a, size_t blocks);
void bi_mul_pow10(BigInt* a, int n);
int bi_cmp_mag(const BigInt* a, const BigInt* b);
BigInt* bi_add_mag(const BigInt* a, const BigInt* b);
BigInt* bi_sub_mag(const BigInt* a, const BigInt* b);
BigInt* bi_add(const BigInt* a, const BigInt* b);
BigInt* bi_sub(const BigInt* a, const BigInt* b);
BigInt* bi_mul(const BigInt* a, const BigInt* b);
BigInt* bi_mul_single(const BigInt* a, uint32_t multiplier);
void bi_mul_single_inplace(BigInt* buffer, const BigInt* a, uint32_t multiplier);
BigInt* bi_div_mag(const BigInt* a, const BigInt* b);

BigDec* bd_new();
void bd_free(BigDec* a);
BigDec* bd_copy(const BigDec* a);
void bd_align(BigDec* a, BigDec* b);
BigDec* bd_add(BigDec* a, BigDec* b);
BigDec* bd_sub(BigDec* a, BigDec* b);
BigDec* bd_mul(BigDec* a, BigDec* b);
BigDec* bd_div(BigDec* a, BigDec* b);
void bd_trim(BigDec* a);
BigDec* bd_parse(const char* str);
void bd_print(BigDec* a);
BigDec* bd_sqrt(BigDec *n);
BigDec* bd_cbrt(BigDec *n);
BigDec* bd_sin(BigDec *x);
BigDec* bd_cos(BigDec *x);
BigDec* execute_math_function(const char* func_name, BigDec* arg);

void next_token();
BigDec* parse_factor();
BigDec* parse_term();
BigDec* parse_expr();

// ============================================================================
// 错误处理实现 (Error Handling)
// ============================================================================
void set_error(const char* msg) {
  if (!g_err_flag) {
    fprintf(stderr, "Error: %s\n", msg);
    
    // 带箭头的高亮错误指示
    if (expr_start) {
      fprintf(stderr, "    %s\n", expr_start);
      fprintf(stderr, "    ");
      for (size_t i = 0; i < curr_tok.pos; i++) {
        fprintf(stderr, " ");
      }
      fprintf(stderr, "\033[1;31m^\033[0m\n");
    }
    g_err_flag = 1;
  }
}

// ============================================================================
// 大整数模块实现 (BigInt Implementation)
// ============================================================================
BigInt* bi_new() {
  BigInt* res = (BigInt*)malloc(sizeof(BigInt));
  if (!res) { set_error("Memory allocation failed (OOM)"); return NULL; }
  
  res->sign = 1;
  res->len = 1;
  res->dig = (uint32_t*)malloc(sizeof(uint32_t));
  if (!res->dig) { 
      free(res); 
      set_error("Memory allocation failed (OOM)"); 
      return NULL; 
  }
  res->dig[0] = 0;
  return res;
}

void bi_free(BigInt* a) {
  if (a) {
    if (a->dig) free(a->dig);
    free(a);
  }
}

void bi_trim(BigInt* a) {
  // 去除高位多余的零
  while (a->len > 1 && a->dig[a->len - 1] == 0) {
    a->len--;
  }
  // 防止出现负零
  if (a->len == 1 && a->dig[0] == 0) {
    a->sign = 1;
  }
}

BigInt* bi_copy(const BigInt* a) {
  if (!a) return NULL;
  BigInt* res = (BigInt*)malloc(sizeof(BigInt));
  if (!res) { set_error("Memory allocation failed (OOM)"); return NULL; }

  res->sign = a->sign;
  res->len = a->len;
  res->dig = (uint32_t*)malloc(a->len * sizeof(uint32_t));
  if (!res->dig) {
      free(res);
      set_error("Memory allocation failed (OOM)");
      return NULL;
  }
  memcpy(res->dig, a->dig, a->len * sizeof(uint32_t));
  return res;
}

//实现左移n个区块
void bi_shift_blocks_left(BigInt* a, size_t blocks) {
  if (blocks == 0) return;
  if (a->len == 1 && a->dig[0] == 0) return;
  uint32_t* new_dig = (uint32_t*)calloc(a->len + blocks, sizeof(uint32_t));
  memcpy(new_dig + blocks, a->dig, a->len * sizeof(uint32_t));
  free(a->dig);
  a->dig = new_dig;
  a->len += blocks;
}

//实现乘以10的n次方
void bi_mul_pow10(BigInt* a, int n) {
  if (n <= 0 || (a->len == 1 && a->dig[0] == 0)) return;
  int blocks = n / 9;
  int rem = n % 9;

  bi_shift_blocks_left(a, blocks);

  if (rem > 0) {
    uint32_t multiplier = 1;
    for (int i = 0; i < rem; i++) multiplier *= 10;

    uint32_t carry = 0;
    for (size_t i = 0; i < a->len; i++) {
      uint64_t prod = (uint64_t)a->dig[i] * multiplier + carry;
      a->dig[i] = prod % BASE;
      carry = prod / BASE;
    }
    if (carry > 0) {
      uint32_t* new_dig = (uint32_t*)malloc((a->len + 1) * sizeof(uint32_t));
      if (!new_dig) {
        set_error("Memory allocation failed during pow10 shift (OOM)");
        return; 
      }
      memcpy(new_dig, a->dig, a->len * sizeof(uint32_t));
      free(a->dig);
      a->dig = new_dig;
      a->dig[a->len] = carry;
      a->len++;
    }
  }
}

//大整数的绝对值比较
int bi_cmp_mag(const BigInt* a, const BigInt* b) {
  if (a->len != b->len) return a->len > b->len ? 1 : -1;
  for (uint32_t i = a->len; i-- > 0;) {
    if (a->dig[i] != b->dig[i]) return a->dig[i] > b->dig[i] ? 1 : -1;
  }
  return 0;
}

//大整数的绝对值加法
BigInt* bi_add_mag(const BigInt* a, const BigInt* b) {
  BigInt* res = (BigInt*)malloc(sizeof(BigInt));
  res->sign = 1;
  size_t max_len = a->len > b->len ? a->len : b->len;
  res->dig = (uint32_t*)malloc((max_len + 1) * sizeof(uint32_t));
  uint32_t carry = 0;
  for (size_t i = 0; i < max_len; i++) {
    uint32_t d1 = i < a->len ? a->dig[i] : 0;
    uint32_t d2 = i < b->len ? b->dig[i] : 0;
    uint32_t sum = d1 + d2 + carry;
    res->dig[i] = sum % BASE;
    carry = sum / BASE;
  }
  if (carry > 0) {
    res->dig[max_len] = carry;
    res->len = max_len + 1;
  } else {
    res->len = max_len;
  }
  bi_trim(res);
  return res;
}

BigInt* bi_sub_mag(const BigInt* a, const BigInt* b) {
  BigInt* res = (BigInt*)malloc(sizeof(BigInt));
  res->sign = 1;
  res->dig = (uint32_t*)malloc(a->len * sizeof(uint32_t));
  uint32_t borrow = 0;
  for (size_t i = 0; i < a->len; i++) {
    uint32_t d1 = a->dig[i];
    uint32_t d2 = i < b->len ? b->dig[i] : 0;
    int64_t sub = (int64_t)d1 - d2 - borrow;
    if (sub < 0) {
      sub += BASE;
      borrow = 1;
    } else {
      borrow = 0;
    }
    res->dig[i] = (uint32_t)sub;
  }
  res->len = a->len;
  bi_trim(res);
  return res;
}

BigInt* bi_add(const BigInt* a, const BigInt* b) {
  if (a->sign == b->sign) {
    BigInt* res = bi_add_mag(a, b);
    res->sign = a->sign;
    return res;
  }
  int cmp = bi_cmp_mag(a, b);
  if (cmp >= 0) {
    BigInt* res = bi_sub_mag(a, b);
    res->sign = a->sign;
    return res;
  } else {
    BigInt* res = bi_sub_mag(b, a);
    res->sign = b->sign;
    return res;
  }
}

BigInt* bi_sub(const BigInt* a, const BigInt* b) {
  BigInt* b_neg = bi_copy(b);
  b_neg->sign = -b->sign;
  BigInt* res = bi_add(a, b_neg);
  bi_free(b_neg);
  return res;
}

//FFT乘法
static inline Complex c_add(Complex a, Complex b) {
  return (Complex){a.r + b.r, a.i + b.i};
}
static inline Complex c_sub(Complex a, Complex b) {
  return (Complex){a.r - b.r, a.i - b.i};
}
static inline Complex c_mul(Complex a, Complex b) {
  return (Complex){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

static void fft(Complex* a, int n, int invert) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      Complex temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    long double angle = 2.0L * M_PI_L / len * (invert ? -1 : 1);
    Complex wlen = {cosl(angle), sinl(angle)};
    for (int i = 0; i < n; i += len) {
      Complex w = {1, 0};
      for (int j = 0; j < len / 2; j++) {
        Complex u = a[i + j];
        Complex v = c_mul(a[i + j + len / 2], w);
        a[i + j] = c_add(u, v);
        a[i + j + len / 2] = c_sub(u, v);
        Complex next_w;
        next_w.r = w.r * wlen.r - w.i * wlen.i;
        next_w.i = w.r * wlen.i + w.i * wlen.r;
        w = next_w;
      }
    }
  }
  if (invert) {
    for (int i = 0; i < n; i++) {
      a[i].r /= n;
      a[i].i /= n;
    }
  }
}

//O(N)单区块乘法
BigInt* bi_mul_single(const BigInt* a, uint32_t multiplier) {
  BigInt* res = bi_new();
  if (multiplier == 0 || (a->len == 1 && a->dig[0] == 0)) {
    return res;
  }
  free(res->dig);
  res->dig = (uint32_t*)malloc((a->len + 1) * sizeof(uint32_t));

  uint32_t carry = 0;
  for (size_t i = 0; i < a->len; i++) {
    uint64_t prod = (uint64_t)a->dig[i] * multiplier + carry;
    res->dig[i] = prod % BASE;
    carry = prod / BASE;
  }

  if (carry > 0) {
    res->dig[a->len] = carry;
    res->len = a->len + 1;
  } else {
    res->len = a->len;
  }
  res->sign = a->sign;
  return res;
}

// 原位内存单区块乘法，极大优化试商过程的内存压力
void bi_mul_single_inplace(BigInt* buffer, const BigInt* a, uint32_t multiplier) {
  if (multiplier == 0 || (a->len == 1 && a->dig[0] == 0)) {
    buffer->len = 1;
    buffer->dig[0] = 0;
    return;
  }
  uint32_t carry = 0;
  for (size_t i = 0; i < a->len; i++) {
    uint64_t prod = (uint64_t)a->dig[i] * multiplier + carry;
    buffer->dig[i] = prod % BASE;
    carry = prod / BASE;
  }
  if (carry > 0) {
    buffer->dig[a->len] = carry;
    buffer->len = a->len + 1;
  } else {
    buffer->len = a->len;
  }
}

BigInt* bi_mul(const BigInt* a, const BigInt* b) {
  BigInt* res = (BigInt*)malloc(sizeof(BigInt));
  res->sign = a->sign * b->sign;
  res->len = a->len + b->len;

  // 短数字使用常规乘法优化开销
  if (a->len < 64 || b->len < 64) {
    res->dig = (uint32_t*)calloc(res->len, sizeof(uint32_t));
    for (size_t i = 0; i < a->len; i++) {
      uint32_t carry = 0;
      for (size_t j = 0; j < b->len; j++) {
        uint64_t prod = (uint64_t)a->dig[i] * b->dig[j] + res->dig[i + j] + carry;
        res->dig[i + j] = prod % BASE;
        carry = prod / BASE;
      }
      res->dig[i + b->len] += carry;
    }
    bi_trim(res);
    return res;
  }

  size_t n = 1;
  while (n < 3 * (size_t)a->len + 3 * (size_t)b->len) {
    n <<= 1;
  }
  Complex* fa = (Complex*)calloc(n, sizeof(Complex));
  Complex* fb = (Complex*)calloc(n, sizeof(Complex));
  if (!fa || !fb) {
    if (fa) free(fa);
    if (fb) free(fb);
    set_error("Memory allocation failed during FFT (OOM)");
    free(res);
    return bi_new(); 
  }
  for (size_t i = 0; i < a->len; i++) {
    fa[3 * i].r = a->dig[i] % 1000;
    fa[3 * i + 1].r = (a->dig[i] / 1000) % 1000;
    fa[3 * i + 2].r = a->dig[i] / 1000000;
  }
  for (size_t i = 0; i < b->len; i++) {
    fb[3 * i].r = b->dig[i] % 1000;
    fb[3 * i + 1].r = (b->dig[i] / 1000) % 1000;
    fb[3 * i + 2].r = b->dig[i] / 1000000;
  }

  fft(fa, n, 0);
  fft(fb, n, 0);

  for (size_t i = 0; i < n; i++) {
    fa[i] = c_mul(fa[i], fb[i]);
  }

  fft(fa, n, 1);

  res->dig = (uint32_t*)calloc(res->len, sizeof(uint32_t));
  uint64_t carry = 0;
  for (size_t i = 0; i < n; i++) {
    int64_t coeff = (int64_t)roundl(fa[i].r);
    uint64_t val = (uint64_t)coeff + carry;
    carry = val / 1000;
    val %= 1000;

    int idx = i / 3;
    int part = i % 3;
    if ((size_t)idx < res->len) {
      if (part == 0) res->dig[idx] += (uint32_t)val;
      else if (part == 1) res->dig[idx] += (uint32_t)(val * 1000);
      else if (part == 2) res->dig[idx] += (uint32_t)(val * 1000000);
    }
  }
  
  size_t final_len = res->len;
  while (carry > 0) {
    if (final_len >= res->len) {
      size_t new_len = res->len + 1;
      uint32_t* tmp = (uint32_t*)realloc(res->dig, new_len * sizeof(uint32_t));
      if (!tmp) {
        set_error("Memory allocation failed during carry propagation");
        break;
      }
      res->dig = tmp;
      res->len = new_len;
      res->dig[res->len - 1] = 0;
    }
    res->dig[final_len] += (uint32_t)(carry % 1000);
    carry /= 1000;
    final_len++;
  }

  free(fa);
  free(fb);
  bi_trim(res);
  return res;
}

// 二分试商法求绝对值除法
BigInt* bi_div_mag(const BigInt* a, const BigInt* b) {
  if (b->len == 1 && b->dig[0] == 0) {
    set_error("Division by zero");
    return bi_new();
  }

  BigInt* res = bi_new();
  free(res->dig);
  res->dig = (uint32_t*)calloc(a->len, sizeof(uint32_t));
  res->len = a->len;

  BigInt* rem = bi_new();
  BigInt* prod_buffer = bi_new();
  free(prod_buffer->dig);
  prod_buffer->dig = (uint32_t*)malloc((b->len + 1) * sizeof(uint32_t));

  for (uint32_t i = a->len; i-- > 0;) {
    bi_shift_blocks_left(rem, 1);
    rem->dig[0] = a->dig[i];

    uint32_t L = 0, R = BASE - 1, ans = 0;
    while (L <= R) {
      uint32_t mid = L + (R - L) / 2;
      bi_mul_single_inplace(prod_buffer, b, mid);

      if (bi_cmp_mag(prod_buffer, rem) <= 0) {
        ans = mid;
        L = mid + 1;
      } else {
        R = mid - 1;
      }
    }
    res->dig[i] = ans;

    bi_mul_single_inplace(prod_buffer, b, ans);
    BigInt* temp = bi_sub_mag(rem, prod_buffer);
    bi_free(rem);
    rem = temp;
  }

  bi_free(prod_buffer);
  bi_free(rem);
  bi_trim(res);
  return res;
}

// ============================================================================
// 高精度小数模块实现 (BigDec Implementation)
// ============================================================================
BigDec* bd_new() {
  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_new();
  res->scale = 0;
  return res;
}

void bd_free(BigDec* a) {
  if (a) {
    bi_free(a->val);
    free(a);
  }
}

BigDec* bd_copy(const BigDec* a) {
  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_copy(a->val);
  res->scale = a->scale;
  return res;
}

// 小数点对齐操作
void bd_align(BigDec* a, BigDec* b) {
  if (a->scale < b->scale) {
    int diff = b->scale - a->scale;
    bi_mul_pow10(a->val, diff);
    a->scale = b->scale;
  } else if (a->scale > b->scale) {
    int diff = a->scale - b->scale;
    bi_mul_pow10(b->val, diff);
    b->scale = a->scale;
  }
}

BigDec* bd_add(BigDec* a, BigDec* b) {
  BigDec* ac = bd_copy(a);
  BigDec* bc = bd_copy(b);
  bd_align(ac, bc);
  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_add(ac->val, bc->val);
  res->scale = ac->scale;
  bd_free(ac);
  bd_free(bc);
  return res;
}

BigDec* bd_sub(BigDec* a, BigDec* b) {
  BigDec* ac = bd_copy(a);
  BigDec* bc = bd_copy(b);
  bd_align(ac, bc);
  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_sub(ac->val, bc->val);
  res->scale = ac->scale;
  bd_free(ac);
  bd_free(bc);
  return res;
}

BigDec* bd_mul(BigDec* a, BigDec* b) {
  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_mul(a->val, b->val);
  res->scale = a->scale + b->scale;
  return res;
}

BigDec* bd_div(BigDec* a, BigDec* b) {
  int shift = DECIMAL_PLACES + b->scale - a->scale;
  BigInt* num = bi_copy(a->val);
  if (shift > 0) bi_mul_pow10(num, shift);

  BigInt* den = bi_copy(b->val);
  if (shift < 0) bi_mul_pow10(den, -shift);

  if (den->len == 1 && den->dig[0] == 0) {
    set_error("Division by zero");
    bi_free(num);
    bi_free(den);
    return bd_new();
  }

  BigDec* res = (BigDec*)malloc(sizeof(BigDec));
  res->val = bi_div_mag(num, den);
  res->val->sign = a->val->sign * b->val->sign;
  res->scale = DECIMAL_PLACES;

  bi_free(num);
  bi_free(den);
  return res;
}


BigDec* bd_sqrt(BigDec *n) {
    if (n->val->sign == -1) {
        set_error("Math Error: Square root of a negative number");
        return bd_new();
    }
    if (n->val->len == 1 && n->val->dig[0] == 0) {
        return bd_new();
    }

    BigDec *guess = bd_parse("1.0");
    BigDec *two = bd_parse("2.0");

    // 牛顿迭代: x = 0.5 * (x + n / x)
    for (int i = 0; i < 20; i++) {
        BigDec *div_res = bd_div(n, guess);          // n / x
        BigDec *sum_res = bd_add(guess, div_res);    // x + n / x
        BigDec *next_guess = bd_div(sum_res, two);  // (x + n / x) / 2

        bd_free(div_res);
        bd_free(sum_res);
        bd_free(guess);
        
        guess = next_guess;
    }
    
    bd_free(two);
    return guess;
}

BigDec* bd_cbrt(BigDec *n) {
    if (n->val->len == 1 && n->val->dig[0] == 0) {
        return bd_new();
    }

    int is_negative = (n->val->sign == -1);
    BigDec *abs_n = bd_copy(n);
    abs_n->val->sign = 1;

    BigDec *guess = bd_parse("1.0"); 
    BigDec *two = bd_parse("2.0");
    BigDec *three = bd_parse("3.0");

    for (int i = 0; i < 25; i++) {
        BigDec *x_sq = bd_mul(guess, guess);
        BigDec *div_res = bd_div(abs_n, x_sq);
        BigDec *two_x = bd_mul(guess, two);
        BigDec *sum_res = bd_add(two_x, div_res);
        BigDec *next_guess = bd_div(sum_res, three);

        bd_free(x_sq);
        bd_free(div_res);
        bd_free(two_x);
        bd_free(sum_res);
        bd_free(guess);
        
        guess = next_guess;
    }
    
    bd_free(two);
    bd_free(three);
    bd_free(abs_n);

    if (is_negative) {
        guess->val->sign = -1;
    }
    
    return guess;
}
BigDec* bd_trunc(BigDec *a) {
    char *out = (char*)malloc(a->val->len * 9 + 2);
    int pos = 0;
    pos += sprintf(out + pos, "%u", a->val->dig[a->val->len - 1]);
    for (int i = a->val->len - 2; i >= 0; i--) {
        pos += sprintf(out + pos, "%09u", a->val->dig[i]);
    }
    out[pos] = '\0';

    int total_digits = pos;
    BigDec *res;
    if (a->scale >= total_digits) {
        res = bd_parse("0");
    } else {
        int int_len = total_digits - a->scale;
        char *int_str = (char*)malloc(int_len + 2);
        int idx = 0;
        if (a->val->sign == -1) int_str[idx++] = '-';
        strncpy(int_str + idx, out, int_len);
        int_str[idx + int_len] = '\0';
        res = bd_parse(int_str);
        free(int_str);
    }
    free(out);
    return res;
}
BigDec* bd_sin(BigDec *x) {
    BigDec *two_pi = bd_parse("6.28318530717958647692528676655900576839433879875021");
    BigDec *one = bd_parse("1.0");
    BigDec *div_res = bd_div(x, two_pi);
    BigDec *k = bd_trunc(div_res); 
    BigDec *k_times_two_pi = bd_mul(k, two_pi);
    BigDec *x_reduced_raw = bd_sub(x, k_times_two_pi); 
    BigDec *x_reduced = bd_div(x_reduced_raw, one); 

    bd_free(div_res); bd_free(k); bd_free(k_times_two_pi); bd_free(x_reduced_raw); bd_free(two_pi);

    BigDec *sum = bd_copy(x_reduced);   
    BigDec *term = bd_copy(x_reduced);  

    BigDec *x_sq_raw = bd_mul(x_reduced, x_reduced);
    BigDec *x_sq = bd_div(x_sq_raw, one);
    bd_free(x_sq_raw);

    for (int n = 1; n <= 100; n++) { 
        int denominator_int = (2 * n) * (2 * n + 1);
        char buf[32];
        sprintf(buf, "%d.0", denominator_int);
        BigDec *divisor = bd_parse(buf);

        BigDec *temp_mul = bd_mul(term, x_sq);
        BigDec *next_term = bd_div(temp_mul, divisor);
        next_term->val->sign = -next_term->val->sign;

        BigDec *next_sum = bd_add(sum, next_term);

        bd_free(divisor); bd_free(temp_mul); bd_free(term); bd_free(sum);
        term = next_term;
        sum = next_sum;

        if (term->val->len == 1 && term->val->dig[0] == 0) {
            break; 
        }
    }

    bd_free(x_sq); bd_free(one); bd_free(term); bd_free(x_reduced);
    return sum;
}
BigDec* bd_cos(BigDec *x) {
    BigDec *pi_over_two = bd_parse("1.57079632679489661923132169163975144209858469968755");
    
    BigDec *x_plus_pi_half = bd_add(x, pi_over_two);
    
    BigDec *res = bd_sin(x_plus_pi_half);
    
    bd_free(pi_over_two);
    bd_free(x_plus_pi_half);
    
    return res;
}

BigDec* execute_math_function(const char* func_name, BigDec* arg) {
    if (g_err_flag) return bd_new();
    
    if (strcmp(func_name, "sqrt") == 0) {
        return bd_sqrt(arg);
    } 
    else if (strcmp(func_name, "cbrt") == 0) {
        return bd_cbrt(arg);
    }
    else if (strcmp(func_name, "sin") == 0) {
        return bd_sin(arg);
    }
    else if (strcmp(func_name, "cos") == 0) {
        return bd_cos(arg);
    }
    
    char err_msg[100];
    sprintf(err_msg, "Unknown math function: %s", func_name);
    set_error(err_msg);
    return bd_new();
}

void bd_trim(BigDec* a) {
  while (a->scale > 0) {
    if (a->val->dig[0] % 10 != 0) break;
    uint32_t rem = 0;
    for (uint32_t i = a->val->len; i-- > 0;) {
      uint64_t current = (uint64_t)rem * BASE + a->val->dig[i];
      a->val->dig[i] = current / 10;
      rem = current % 10;
    }
    a->scale--;
    bi_trim(a->val);
  }
  if (a->val->len == 1 && a->val->dig[0] == 0) a->scale = 0;
}

BigDec* bd_parse(const char* str) {
  int sign = 1;
  if (*str == '-') {
    sign = -1;
    str++;
  } else if (*str == '+') {
    str++;
  }

  char* clean_str = (char*)malloc(MAX_BUF_SIZE);
  if (!clean_str) {
    set_error("Memory allocation failed for parsing");
    return bd_new();
  }
  
  int clean_len = 0;
  int scale = 0;
  int has_dot = 0;
  int exp_val = 0;

  const char* p = str;
  while (*p) {
    if (*p == 'e' || *p == 'E') {
    char* end;
    long long tmp = strtoll(p + 1, &end, 10);
    if (tmp > 1000000 || tmp < -1000000) {
        set_error("Exponent too large");
        free(clean_str);
        return bd_new();
    }
    exp_val = (int)tmp;
    if (end == p + 1) {
        set_error("Invalid exponent");
        free(clean_str);
        return bd_new();
    }
    break;
    } else if (*p == '.') {
      has_dot = 1;
    } else if (isdigit(*p)) {
      if (clean_len >= MAX_BUF_SIZE - 1) {
        set_error("Number too long");
        free(clean_str);
        return bd_new();
      }
      clean_str[clean_len++] = *p;
      if (has_dot) scale++;
    } else {
      break;
    }
    p++;
  }
  clean_str[clean_len] = '\0';

  BigDec* res = bd_new();
  res->val->sign = sign;

  scale -= exp_val;
  if (scale < 0) {
    res->scale = 0;
  } else {
    res->scale = scale;
  }

  if (clean_len == 0) {
    free(clean_str);
    return res;
  }

  int blocks = (clean_len + 8) / 9;
  free(res->val->dig);
  res->val->dig = (uint32_t*)calloc(blocks, sizeof(uint32_t));
  res->val->len = blocks;

  for (int i = 0; i < blocks; i++) {
    int end = clean_len - i * 9;
    int start = end - 9;
    if (start < 0) start = 0;
    uint32_t chunk_val = 0;
    for (int j = start; j < end; j++) {
      chunk_val = chunk_val * 10 + (clean_str[j] - '0');
    }
    res->val->dig[i] = chunk_val;
  }
  bi_trim(res->val);

  if (scale < 0) bi_mul_pow10(res->val, -scale);
  free(clean_str);
  return res;
}

void bd_print(BigDec* a) {
  if (g_err_flag) return;
  bd_trim(a);
  
  if (a->val->sign == -1 && !(a->val->len == 1 && a->val->dig[0] == 0)) {
    printf("-");
  }

  char* out = (char*)malloc((size_t)a->val->len * 9 + 1);
  if (!out) { set_error("Memory allocation failed in bd_print"); return; }

  size_t pos = 0;

  pos += sprintf(out + pos, "%" PRIu32, a->val->dig[a->val->len - 1]);

  //使用 a->val->len - 1 作为起点，防止最高位被重复打印
  for (uint32_t i = a->val->len - 1; i-- > 0; ) {
    pos += snprintf(out + pos, (size_t)a->val->len * 9 + 1 - pos,
                "%09" PRIu32, a->val->dig[i]);
  }
  out[pos] = '\0';

  size_t total_digits = pos;
  size_t scale_size = (size_t)a->scale; 
  long long exp_val = (long long)total_digits - 1 - (long long)scale_size;

  //大于等于 10^30 或 小于等于 10^-10 时，启用科学计数法输出
  if (exp_val >= 30 || exp_val <= -10) {
    printf("%c", out[0]); 
    size_t last_non_zero = total_digits - 1;
    while (last_non_zero > 0 && out[last_non_zero] == '0') {
      last_non_zero--;
    }

    if (last_non_zero > 0) {
      printf(".");
      for (size_t i = 1; i <= last_non_zero; i++) {
        printf("%c", out[i]);
      }
    }
    
    printf("e%lld", exp_val);
    
  } else {
    if (scale_size >= total_digits) {
      printf("0.");
      for (size_t i = 0; i < scale_size - total_digits; i++) printf("0");
      printf("%s", out);
    } else {
      size_t dot_pos = total_digits - scale_size;
      for (size_t i = 0; i < total_digits; i++) {
      if (i == dot_pos) printf(".");
      printf("%c", out[i]);
    }
  }}
  printf("\n");
  free(out);
}

// ============================================================================
// 表达式解析模块 (Expression Parser)
// ============================================================================

void next_token() {
  if (g_err_flag) return;

  while (isspace((unsigned char)*expr_ptr)) expr_ptr++;
  
  curr_tok.pos = expr_ptr - expr_start;
  
  if (*expr_ptr == '\0') { curr_tok.type = TOK_EOF; return; }
  if (*expr_ptr == '+') { curr_tok.type = TOK_PLUS; expr_ptr++; return; }
  if (*expr_ptr == '-') { curr_tok.type = TOK_MINUS; expr_ptr++; return; }
  if (*expr_ptr == '*') { curr_tok.type = TOK_MUL; expr_ptr++; return; }
  if (*expr_ptr == '/') { curr_tok.type = TOK_DIV; expr_ptr++; return; }
  if (*expr_ptr == '(') { curr_tok.type = TOK_LPAREN; expr_ptr++; return; }
  if (*expr_ptr == ')') { curr_tok.type = TOK_RPAREN; expr_ptr++; return; }
  
  if (isalpha(*expr_ptr)) {
        int i = 0;
        while (isalpha(*expr_ptr)) {
            if (i >= sizeof(curr_tok.val_str) - 1) {
                set_error("Function name too long");
                break;
            }
            curr_tok.val_str[i++] = *expr_ptr++;
        }
        curr_tok.val_str[i] = '\0';
        curr_tok.type = TOK_FUNC;
        return;
    }

  if (isdigit((unsigned char)*expr_ptr) || *expr_ptr == '.') {
    int i = 0, dot_count = 0, has_e = 0;
    int has_digit = 0;

    while (isdigit((unsigned char)*expr_ptr) || *expr_ptr == '.' ||
           (!has_e && (*expr_ptr == 'e' || *expr_ptr == 'E'))) {
      
      if ((size_t)i >= sizeof(curr_tok.val_str) - 1) {
        set_error("Number too long (Buffer Overflow)");
        break;
      }

      if (isdigit((unsigned char)*expr_ptr)) {
        has_digit = 1;
        curr_tok.val_str[i++] = *expr_ptr++;
        continue;
      }

      if (*expr_ptr == '.') {
        if (has_e) {
          set_error("Invalid number: decimal point in exponent");
          curr_tok.type = TOK_EOF;
          return;
        }
        dot_count++;
        if (dot_count > 1) {
          set_error("Multiple decimal points");
          curr_tok.type = TOK_EOF;
          return;
        }
        curr_tok.val_str[i++] = *expr_ptr++;
        continue;
      }

      if (*expr_ptr == 'e' || *expr_ptr == 'E') {
        if (!has_digit) {
          set_error("Invalid number: missing base before 'e'");
          curr_tok.type = TOK_EOF;
          return;
        }
        has_e = 1;
        curr_tok.val_str[i++] = *expr_ptr++;
        if (*expr_ptr == '+' || *expr_ptr == '-') {
          curr_tok.val_str[i++] = *expr_ptr++;
        }
        if (!isdigit((unsigned char)*expr_ptr)) {
          set_error("Invalid number: missing exponent after 'e'");
          curr_tok.type = TOK_EOF;
          return;
        }
        continue;
      }
    }
    curr_tok.val_str[i] = '\0';

    if (!has_digit) {
      set_error("Invalid number: isolated decimal point");
      curr_tok.type = TOK_EOF;
      return;
    }
    curr_tok.type = TOK_NUM;
    return;
  }

  char err_msg[50];
  sprintf(err_msg, "Unknown character '%c'", *expr_ptr);
  set_error(err_msg);
}

BigDec* parse_factor() {
  if (g_err_flag) return bd_new();

  if (curr_tok.type == TOK_FUNC) {
        char func_name[MAX_BUF_SIZE];
        strcpy(func_name, curr_tok.val_str);
        next_token();

        if (curr_tok.type != TOK_LPAREN) {
            set_error("Variables are not supported, or missing '(' for function");
            return bd_new();
        }
        next_token();

        BigDec *arg = parse_expr(); 

        if (curr_tok.type != TOK_RPAREN) {
            set_error("Expected ')' after function argument");
            bd_free(arg);
            return bd_new();
        }
        next_token();

        BigDec *res = execute_math_function(func_name, arg);
        bd_free(arg);
        return res;
    }
  if (curr_tok.type == TOK_NUM) {
    BigDec* res = bd_parse(curr_tok.val_str);
    next_token();
    return res;
  } else if (curr_tok.type == TOK_LPAREN) {
    next_token();
    if (curr_tok.type == TOK_RPAREN) set_error("Empty parentheses");

    BigDec* res = parse_expr();

    if (curr_tok.type != TOK_RPAREN) {
      set_error("Expected ')'");
      bd_free(res);
      return bd_new();
    }
    next_token();
    return res;
  } else if (curr_tok.type == TOK_MINUS) {
    next_token();
    // 拦截连续一元符号
    if (curr_tok.type == TOK_PLUS || curr_tok.type == TOK_MINUS) {
      set_error("Syntax Error: Multiple consecutive signs are not allowed");
      return bd_new();
    }
    BigDec* res = parse_factor();
    res->val->sign = -res->val->sign;
    return res;
  } else if (curr_tok.type == TOK_PLUS) {
    next_token();
    if (curr_tok.type == TOK_PLUS || curr_tok.type == TOK_MINUS) {
      set_error("Syntax Error: Multiple consecutive signs are not allowed");
      return bd_new();
    }
    return parse_factor();
  }

  set_error("Unexpected token");
  return bd_new();
}
BigDec* parse_term() {
  if (g_err_flag) return bd_new();
  BigDec* res = parse_factor();

  while (curr_tok.type == TOK_MUL || curr_tok.type == TOK_DIV) {
    TokenType op = curr_tok.type;
    next_token();

    //乘除号后面紧跟加减乘除，直接拦截
    if (curr_tok.type == TOK_PLUS || curr_tok.type == TOK_MINUS || 
        curr_tok.type == TOK_MUL || curr_tok.type == TOK_DIV) {
      set_error("Syntax Error: Consecutive operators are not allowed. Use parentheses.");
      bd_free(res);
      return bd_new();
    }

    BigDec* right = parse_factor();

    if (g_err_flag) {
      bd_free(res);
      bd_free(right);
      return bd_new();
    }

    BigDec* temp;
    if (op == TOK_MUL) {
      temp = bd_mul(res, right);
    } else {
      temp = bd_div(res, right);
    }

    bd_free(res);
    bd_free(right);
    res = temp;
  }
  return res;
}

BigDec* parse_expr() {
  if (g_err_flag) return bd_new();
  BigDec* res = parse_term();

  while (curr_tok.type == TOK_PLUS || curr_tok.type == TOK_MINUS) {
    TokenType op = curr_tok.type;
    next_token();

    //加减号后面紧跟加减乘除，直接拦截
    if (curr_tok.type == TOK_PLUS || curr_tok.type == TOK_MINUS || 
        curr_tok.type == TOK_MUL || curr_tok.type == TOK_DIV) {
      set_error("Syntax Error: Multiple consecutive signs are not allowed.");
      bd_free(res);
      return bd_new();
    }

    BigDec* right = parse_term();

    if (g_err_flag) {
      bd_free(res);
      bd_free(right);
      return bd_new();
    }

    BigDec* temp;
    if (op == TOK_PLUS) {
      temp = bd_add(res, right);
    } else {
      temp = bd_sub(res, right);
    }

    bd_free(res);
    bd_free(right);
    res = temp;
  }
  return res;
}

// ============================================================================
// 主程序入口 (Main)
// ============================================================================
int main() {
  printf("CS219 A Simple Calculator\n");

  while (1) {
    g_err_flag = 0;
    char* input = readline("calc> ");
    
    //EOF(Ctrl+D)退出
    if (!input) {
      printf("\n");
      break; 
    }
    if (input[0] != '\0') add_history(input);

    if (strncmp(input, "exit", 4) == 0 || strncmp(input, "quit", 4) == 0) {
      free(input);
      break;
    }

    int is_empty = 1;
    for (int i = 0; input[i]; i++) {
      if (!isspace((unsigned char)input[i])) {
        is_empty = 0;
        break;
      }
    }
    
    if (is_empty) {
      free(input);
      continue;
    }
    
    expr_start = input;
    expr_ptr = input;
    next_token();
    
    BigDec* res = parse_expr();

    if (curr_tok.type != TOK_EOF) {
      set_error("Syntax Error. Unexpected input at the end.");
    }
    if (!g_err_flag) {
      printf("= ");
      bd_print(res);
    }

    bd_free(res);
    free(input);
  }
  return 0;
}