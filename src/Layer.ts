/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

export class Layer {
  public weights: number[][];
  public bias: number[];

  constructor(
    public inSize: number,
    public outSize: number,
  ) {
    const std = Math.sqrt(2 / inSize);
    this.weights = Array.from({ length: inSize }, () =>
      Array.from({ length: outSize }, () => (Math.random() * 2 - 1) * std),
    );
    this.bias = new Array(outSize).fill(0.01);
  }
}
