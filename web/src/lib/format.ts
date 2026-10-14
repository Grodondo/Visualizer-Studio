export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** Fixed decimals with a real minus sign; tiny values print as 0 instead of "-0.00". */
export function fmt(v: number, decimals = 2): string {
  const rounded = Math.abs(v) < 0.5 * 10 ** -decimals ? 0 : v;
  return rounded.toFixed(decimals).replace("-", "\u2212");
}

export const prefersReducedMotion = (): boolean => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
