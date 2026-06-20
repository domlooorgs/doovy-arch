/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

export const relu = (x: number) => Math.max(0, x);
export const reluDerivative = (x: number) => (x > 0 ? 1 : 0);
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
