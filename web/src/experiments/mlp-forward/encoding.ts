// Visual encodings shared by every view of this experiment. Colors live in ui/theme.css.

/** |weight| that maps to the thickest line / strongest matrix cell. */
export const WEIGHT_MAX = 4;
/** |w * a| that maps to a fully opaque line in "signal" mode. */
export const SIGNAL_MAX = 3;
/** Range of the weight and bias sliders. */
export const PARAM_SLIDER = { min: -8, max: 8, step: 0.05 };
/** Layers per second during Play. */
export const REVEAL_SPEED = 0.8;

/** One meaning everywhere: blue = positive number, orange = negative number. */
export const signColor = (v: number): string => (v >= 0 ? "var(--pos)" : "var(--neg)");

export const magnitude = (v: number, max: number): number => Math.min(1, Math.abs(v) / max);

/** How much of layer l has been revealed at a given stage (layer 0 is always fully shown). */
export const reveal = (stage: number, layer: number): number => (layer === 0 ? 1 : Math.min(1, Math.max(0, stage - (layer - 1))));
