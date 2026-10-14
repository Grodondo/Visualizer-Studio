/** Single source of truth for one experiment: controls and views change it, then everything redraws. */
export class Store<S> {
  private state: S;
  private listeners = new Set<(s: S) => void>();

  constructor(initial: S) {
    this.state = initial;
  }

  get(): S {
    return this.state;
  }

  set(next: S): void {
    this.state = next;
    for (const fn of this.listeners) fn(next);
  }

  update(fn: (s: S) => S): void {
    this.set(fn(this.state));
  }

  subscribe(fn: (s: S) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
