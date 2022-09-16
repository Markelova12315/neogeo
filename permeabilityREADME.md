# neogeo/permeabillity
#include <cstdio> #include <cmath> #include <cstring> #include <cassert> #include <vector> using namespace std;

#include <gsl/gsl_rng.h> #include <gsl/gsl_randist.h> #include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>

#define NSAMPLE 100000
#define POROSITY_AVERAGE 0.5
#define POROSITY_SIGMA 1.0
#define NDIV_PTY 100
#define NCLASS 5

const double class_thresholds[NCLASS] = {1000, 500, 100, 10, 1};

struct data {
double porosity;	// [dimensionless]
double permeability; // [mD] or [1e-3 * um2]
};

vector<struct data> data;
int nclass[NCLASS][NDIV_PTY];
int npty[NDIV_PTY];
int class0 = NCLASS-1;

void load_data(FILE *f, vector<struct data> &data)
{
char buf[65536];
while (!feof(f)) {
if (fgets(buf, sizeof(buf)/sizeof(buf[0]), f)) {
double x, y;
if (sscanf(buf, "%lf %lf", &x, &y) == 2) {
struct data d; d.porosity = x * 0.01; d.permeability = y; data.push_back(d);
}
}
}
printf("# load %lu points\n", data.size());
}

void classification(const vector<struct data> &data, int nclass[NCLASS][NDIV_PTY], int npty[NDIV_PTY])
{
memset(nclass, 0, sizeof(int)*NCLASS*NDIV_PTY);
memset(npty, 0, sizeof(int)*NDIV_PTY);
for (int i = 0; i < (int) data.size(); i++) {
const struct data &d = data[i]; const double h = 1.0/NDIV_PTY; int index = floor(d.porosity/h); assert(index >= 0); assert(index < NDIV_PTY); npty[index]++;
for (int j = 0; j < NCLASS; j++)
if (d.permeability >= class_thresholds[j]) nclass[j][index]++;
}
 
}

void fit1(const int nmatch[NDIV_PTY], const int ntot[NDIV_PTY], double &a, double &b, double &maxpty)
{
struct function {
struct userdata {
int imax;
double vpty[NDIV_PTY];
};
static double model(const gsl_vector *x, double pty)
{
assert(x->size == 2);
double a = gsl_vector_get(x, 0); double b = gsl_vector_get(x, 1); return 1 - exp(-exp(a*pty - b));
}
static int f(const gsl_vector *x, void *params, gsl_vector *f)
{
assert(x->size == 2);
const struct userdata *userdata = (struct userdata*) params;
assert((int) f->size == userdata->imax);
for (int i = 0; i < userdata->imax; i++) {
const double pty = 100.0 * (i+0.5) * (1.0/NDIV_PTY);
const double v = model(x, pty);
gsl_vector_set(f, i, v - userdata->vpty[i]);
}
return GSL_SUCCESS;
}
};

const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust; gsl_multifit_nlinear_workspace *w;
gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

function::userdata userdata; userdata.imax = -1;
for (int i = 0; i < NDIV_PTY; i++) {
if (ntot[i] > 0)
userdata.vpty[i] = 1.0*nmatch[i]/ntot[i];
else
userdata.vpty[i] = 0;
if (nmatch[i] > 0) userdata.imax = i;
}

gsl_multifit_nlinear_fdf fdf; fdf.f = function::f;
fdf.df = NULL; fdf.fvv = NULL;
fdf.n = userdata.imax; fdf.p = 2;
fdf.params = &userdata;

gsl_vector *x = gsl_vector_alloc(fdf.p); gsl_vector_set(x, 0, 0.1);
gsl_vector_set(x, 1, 1);

const double xtol = 1e-8; const double gtol = 1e-8; const double ftol = 0.0;

maxpty = 100.0 * (userdata.imax-0.5) * (1.0/NDIV_PTY);
 
gsl_vector *xfinal;
if (maxpty >= 0) {
w = gsl_multifit_nlinear_alloc(T, &fdf_params, fdf.n, fdf.p);
gsl_multifit_nlinear_init(x, &fdf, w);
int status, info;
status = gsl_multifit_nlinear_driver(1000, xtol, gtol, ftol, NULL, NULL, &info, w);
int niter = gsl_multifit_nlinear_niter(w);
printf("# fit result %d niter %d\n", status, niter); xfinal = gsl_multifit_nlinear_position(w);
a = gsl_vector_get(xfinal, 0); b = gsl_vector_get(xfinal, 1);
} else {
a = 0;
b = 1e9;
xfinal = gsl_vector_alloc(2); gsl_vector_set(xfinal, 0, a);
gsl_vector_set(xfinal, 1, b);
}
printf("# a %f b %f\n", a, b);
printf("# %8s %10s %10s\n", "Pty", "Data", "Fit");
int imax = userdata.imax*10;
if (imax <= 0) imax = 200;
for (int i = 0; i < imax; i++) {
const double pty = 0.1 * 100.0 * (i+0.5) * (1.0/NDIV_PTY);
if (i % 10 == 0)
printf("%10.3f %10.3f %10.3f\n", pty, userdata.vpty[i/10], function::model(xfinal, pty));
else
printf("%10.3f %10s %10.3f\n", pty, "NA", function::model(xfinal, pty));
}
}

double sqr(double x)
{
return x*x;
}

void generate(const double average_porosity, const double A[NCLASS], const double B[NCLASS], const double maxpty[NCLASS], double pclass[NCLASS])
{
gsl_rng *rng = gsl_rng_alloc(gsl_rng_default); const double sigma1 = average_porosity/2;
const double zeta = log(sqr(average_porosity)/sqrt(sqr(average_porosity)+sqr(sigma1))); const double sigma = sqrt(log(1+sqr(sigma1)/sqr(average_porosity)));
double sum = 0;
for (int j = 0; j < NCLASS; j++) pclass[j] = 0;
for (int i = 0; i < NSAMPLE; i++) {
double pty = gsl_ran_lognormal(rng, zeta, sigma); sum += pty;
for (int j = 0; j < NCLASS; j++) {
double p;
if (pty <= maxpty[j])
p = 1 - exp(-exp(A[j]*pty - B[j]));
else
p = 1;
pclass[j] += p;
}
}
for (int j = 0; j < NCLASS; j++) pclass[j] *= 1.0/NSAMPLE;
}
 
void generate2(const double average_porosity, const double A[NCLASS], const double B[NCLASS], const double maxpty[NCLASS], double pclass[NCLASS])
{
gsl_rng *rng = gsl_rng_alloc(gsl_rng_default); const double sigma1 = average_porosity/2;
const double zeta = log(sqr(average_porosity)/sqrt(sqr(average_porosity)+sqr(sigma1))); const double sigma = sqrt(log(1+sqr(sigma1)/sqr(average_porosity)));
double sum = 0;
for (int j = 0; j < NCLASS; j++) pclass[j] = 0;
for (int i = 0; i < NSAMPLE; i++) {
double pty = gsl_ran_lognormal(rng, zeta, sigma); sum += pty;
double pc[NCLASS];
for (int j = 0; j < NCLASS; j++) {
if (maxpty[j] < 0) pc[j] = 0;
else
if (pty <= maxpty[j])
pc[j] = 1 - exp(-exp(A[j]*pty - B[j]));
 
else

}
 

pc[j] = 1;
 
double t = gsl_rng_uniform(rng); for (int j = 0; j < NCLASS; j++)
if (t <= pc[j])
pclass[j] += 1.0/NSAMPLE;
}
}


void fit2(const double p[NDIV_PTY], double &a1, double &b1, double &c1)
{
struct function {
struct userdata {
double vpty[NDIV_PTY];
};
static double model(const gsl_vector *x, double pty)
{
assert(x->size == 3);
double a1 = gsl_vector_get(x, 0); double b1 = gsl_vector_get(x, 1); double c1 = gsl_vector_get(x, 2);
return 1 - exp(-exp(a1*sqr(pty) + b1*pty + c1));
}
static int f(const gsl_vector *x, void *params, gsl_vector *f)
{
assert(x->size == 3);
const struct userdata *userdata = (struct userdata*) params;
assert((int) f->size == NDIV_PTY);
for (int i = 0; i < NDIV_PTY; i++) {
const double pty = 100.0 * (i+0.5) * (1.0/NDIV_PTY);
const double v = model(x, pty);
gsl_vector_set(f, i, v - userdata->vpty[i]);
}
return GSL_SUCCESS;
}
};

const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust; gsl_multifit_nlinear_workspace *w;
gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
 
function::userdata userdata;
for (int i = 0; i < NDIV_PTY; i++) userdata.vpty[i] = p[i];

gsl_multifit_nlinear_fdf fdf; fdf.f = function::f;
fdf.df = NULL; fdf.fvv = NULL; fdf.n = NDIV_PTY; fdf.p = 3;
fdf.params = &userdata;

gsl_vector *x = gsl_vector_alloc(fdf.p); gsl_vector_set(x, 0, 0.1);
gsl_vector_set(x, 1, 0.1);
gsl_vector_set(x, 2, 1);

const double xtol = 1e-8; const double gtol = 1e-8; const double ftol = 0.0;

w = gsl_multifit_nlinear_alloc(T, &fdf_params, fdf.n, fdf.p);
gsl_multifit_nlinear_init(x, &fdf, w);
int status, info;
status = gsl_multifit_nlinear_driver(1000, xtol, gtol, ftol, NULL, NULL, &info, w);
int niter = gsl_multifit_nlinear_niter(w);
printf("# fit result %d niter %d\n", status, niter); gsl_vector *xfinal = gsl_multifit_nlinear_position(w); a1 = gsl_vector_get(xfinal, 0);
b1 = gsl_vector_get(xfinal, 1); c1 = gsl_vector_get(xfinal, 2);
printf("# a1 %f b1 %f c1 %f\n", a1, b1, c1);
for (int i = 0; i < NDIV_PTY; i++) {
const double pty = 100.0 * (i+0.5) * (1.0/NDIV_PTY);
printf("%10.3f %10.3f %10.3f\n", pty, p[i], function::model(xfinal, pty));
}
}

// https://en.wikipedia.org/wiki/Generalised_logistic_function
void fit2_asymmetric(const double p[NDIV_PTY], double &K, double &C, double &B, double &M, double &nu)
{
struct function {
struct userdata {
double vpty[NDIV_PTY];
};
static double model(const gsl_vector *x, double pty)
{
assert(x->size == 5);
double K = gsl_vector_get(x, 0); double C = gsl_vector_get(x, 1); double B = gsl_vector_get(x, 2); double M = gsl_vector_get(x, 3); double nu = gsl_vector_get(x, 4);
return K / pow(C + exp(-B*(pty-M)), 1/nu);
}
static int f(const gsl_vector *x, void *params, gsl_vector *f)
{
assert(x->size == 5);
const struct userdata *userdata = (struct userdata*) params;
assert((int) f->size == NDIV_PTY);
for (int i = 0; i < NDIV_PTY; i++) {
 
const double pty = 100.0 * (i+0.5) * (1.0/NDIV_PTY);
const double v = model(x, pty);
gsl_vector_set(f, i, v - userdata->vpty[i]);
}
return GSL_SUCCESS;
}
};

const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust; gsl_multifit_nlinear_workspace *w;
gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

function::userdata userdata;
for (int i = 0; i < NDIV_PTY; i++) userdata.vpty[i] = p[i];

gsl_multifit_nlinear_fdf fdf; fdf.f = function::f;
fdf.df = NULL; fdf.fvv = NULL; fdf.n = NDIV_PTY; fdf.p = 5;
fdf.params = &userdata;

gsl_vector *x = gsl_vector_alloc(fdf.p);
gsl_vector_set(x,	0,	1);
gsl_vector_set(x,	1,	1);
gsl_vector_set(x,	2,	0.1);
gsl_vector_set(x,	3,	1);
gsl_vector_set(x,	4,	0.1);

const double xtol = 1e-8; const double gtol = 1e-8; const double ftol = 0.0;

w = gsl_multifit_nlinear_alloc(T, &fdf_params, fdf.n, fdf.p);
gsl_multifit_nlinear_init(x, &fdf, w);
int status, info;
status = gsl_multifit_nlinear_driver(1000, xtol, gtol, ftol, NULL, NULL, &info, w);
int niter = gsl_multifit_nlinear_niter(w);
printf("# fit result %d niter %d\n", status, niter); gsl_vector *xfinal = gsl_multifit_nlinear_position(w); K = gsl_vector_get(xfinal, 0);
C = gsl_vector_get(xfinal, 1); B = gsl_vector_get(xfinal, 2); M = gsl_vector_get(xfinal, 2); nu = gsl_vector_get(xfinal, 2);
printf("# K %g C %g B %g M %g nu %g\n", K, C, B, M, nu);
for (int i = 0; i < NDIV_PTY; i++) {
const double pty = 100.0 * (i+0.5) * (1.0/NDIV_PTY);
printf("%10.3f %10.3f %10.3f\n", pty, p[i], function::model(xfinal, pty));
}
}

void check()
{
for (int i = 0; i < NCLASS-1; i++) {
if (class_thresholds[i] <= class_thresholds[i+1]) { printf("error in class treshold specifications: %d\n", i); exit(2);
}
}
 
}

int main()
{
setvbuf(stdout, NULL, _IOLBF, 0); gsl_rng_env_setup();
check(); load_data(stdin, data);
printf("# input data\n"); classification(data, nclass, npty); printf("# %6s %6s ", "Pty.", "Cnt."); for (int j = 0; j < NCLASS; j++)
printf("%6s%-2d ", "F", j+1);
printf("\n"); int imin = -1; int imax = -1;
for (int i = 0; i < NDIV_PTY; i++) {
if (imin < 0 && npty[i] > 0) imin = i;
if (npty[i] > 0) imax = i;
}
for (int i = imin; i <= imax; i++) { const double h = 1.0 / NDIV_PTY; printf(" %6.3f %6d ", i*h, npty[i]); if (npty[i] > 0) {
for (int j = 0; j < NCLASS; j++) {
printf("%8.3f ", 1.0*nclass[j][i]/npty[i]);
if (nclass[j][i] > 0 && j < class0) class0 = j;
}
}
printf("\n");
}
printf("\n\n");
printf("# begin fit porosity class functions\n");
double A[NCLASS], B[NCLASS], maxpty[NCLASS];
for (int j = 0; j < NCLASS; j++) { fit1(nclass[j], npty, A[j], B[j], maxpty[j]); printf("\n\n");
}
printf("\n\n");
printf("# begin generate simulated permeability distributions\n");
printf("# %4s ", "pty.");
for (int j = 0; j < NCLASS; j++)
printf("pgen%-8d ", j+1);
printf("\n");
double pclass[NCLASS][NDIV_PTY];
for (int i = 0; i < NDIV_PTY; i++) {
double pty = 100.0*(i+0.5)*(1.0/NDIV_PTY); double p[NCLASS];
generate2(pty, A, B, maxpty, p);
printf("%6.2f ", pty);
for (int j = 0; j < NCLASS; j++) {
printf("%12e ", p[j]);
pclass[j][i] = p[j];
}
printf("\n");
}
printf("\n\n");
printf("# begin fit generated class distributions\n");
double K[NCLASS], C[NCLASS], BB[NCLASS], M[NCLASS], nu[NCLASS];
for (int j = 0; j < NCLASS; j++) {
 
fit2_asymmetric(pclass[j], K[j], C[j], BB[j], M[j], nu[j]);
printf("\n\n");
}

printf("# generated cell permeability distributions\n");
printf("# P(%d): permeability from %.0e mD to %.0e mD\n", 0, 0.0, class_thresholds[0]);
for (int j = 0; j < NCLASS-1; j++)
printf("# P(%d): permeability from %.0e mD to %.0e mD\n", j+1, class_thresholds[j], class_thresholds[j+1]);
printf("# %6s ", "Pty.");
for (int j = 0; j < NCLASS; j++)
printf("%9s(%1d) ", "P", j);
printf("\n");
for (int i = 0; i < NDIV_PTY; i++) {
double pty = 100.0*(i+0.5)*(1.0/NDIV_PTY); printf(" %6.2f ", pty);
printf("%12e ", pclass[0][i]);
for (int j = 0; j < NCLASS-1; j++) {
printf("%12e ", fmax(pclass[j+1][i] - pclass[j][i], 0));
}
printf("\n");
}
printf("\n\n");

#define NGROUP 10
double pacc[NCLASS];
int i = 0;
while (i <= NDIV_PTY) {
if (i % NGROUP == 0) {
if (i > 0) {
double pty1 = 100.0*(i-NGROUP)*(1.0/NDIV_PTY); double pty2 = 100.0*i*(1.0/NDIV_PTY); printf("%.0f--%.0f ", pty1, pty2);
for (int j = 0; j < NCLASS; j++)
printf("%12e ", pacc[j]/NGROUP);
printf("\n");
}
memset(pacc, 0, sizeof(pacc));
}
if (i < NDIV_PTY) {
pacc[0] += pclass[0][i];
for (int j = 0; j < NCLASS-1; j++)
pacc[j+1] += fmax(pclass[j+1][i] - pclass[j][i], 0);
} i++;
}
return 0;
}
