import type { AnyExperiment } from "./types";
import { mlpForward } from "../experiments/mlp-forward";

/**
 * The only list of experiments. To add one:
 *   1. create web/src/experiments/<id>/ with an `index.ts` exporting `defineExperiment({...})`
 *   2. import it here and add it to the array
 * The gallery, routing, controls and view switching come from the shell.
 */
export const experiments: AnyExperiment[] = [mlpForward];
