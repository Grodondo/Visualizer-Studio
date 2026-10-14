export type ProvenanceKind = "schematic" | "computed" | "simulated" | "measured";

/** One honest statement about where a part of an experiment comes from. */
export interface ProvenanceNote {
  aspect: string;
  kind: ProvenanceKind;
  note: string;
}

export interface ExperimentMeta {
  id: string;
  title: string;
  topic: string;
  summary: string;
  provenance: ProvenanceNote[];
}

/** What a view may do to the experiment state (e.g. select a neuron). */
export interface ViewHost<S> {
  getState(): S;
  update(fn: (s: S) => S): void;
}

export interface View<S, R> {
  update(state: S, result: R): void;
  destroy(): void;
}

/** One way of drawing the same state and result. Views never compute science. */
export interface ViewDef<S, R, C> {
  id: string;
  label: string;
  /** One sentence: what this style emphasises and what it gives up. */
  tradeoff: string;
  create(container: HTMLElement, host: ViewHost<S>, ctx: C): View<S, R>;
}

interface ControlBase {
  id: string;
  label: string;
  group?: string;
}

export interface SliderSpec<S> extends ControlBase {
  kind: "slider";
  min: number;
  max: number;
  step: number;
  value: number;
  format?: (v: number) => string;
  set: (s: S, v: number) => S;
}

export interface SelectSpec<S> extends ControlBase {
  kind: "select";
  options: { value: string; label: string }[];
  value: string;
  set: (s: S, v: string) => S;
}

export interface ButtonSpec<S> extends ControlBase {
  kind: "button";
  onClick: (s: S) => S;
}

export type ControlSpec<S> = SliderSpec<S> | SelectSpec<S> | ButtonSpec<S>;

export interface Experiment<S, R, C> {
  meta: ExperimentMeta;
  /** Fetch data files. Runs once when the experiment page opens. */
  load(): Promise<C>;
  init(ctx: C): S;
  /** The scientific computation. Pure: same state in, same result out. */
  compute(state: S, ctx: C): R;
  /** Rebuilt after every state change, so values and labels can depend on the state. */
  controls(state: S, result: R, ctx: C): ControlSpec<S>[];
  explain(state: S, result: R, ctx: C): HTMLElement;
  views: ViewDef<S, R, C>[];
  /** Optional time-based behaviour (playback). The shell drives it with requestAnimationFrame. */
  animation?: { active(state: S): boolean; step(state: S, dtMs: number): S };
}

// The registry mixes experiments with different state types.
export type AnyExperiment = Experiment<any, any, any>;

export function defineExperiment<S, R, C>(e: Experiment<S, R, C>): Experiment<S, R, C> {
  return e;
}
