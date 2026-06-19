export class NeuralNetwork {
  public weightsInputHidden: number[][];
  public weightsHiddenOutput: number[][];
  public biasHidden: number[];
  public biasOutput: number[];

  constructor(
    public inputSize: number,
    public hiddenSize: number,
    public outputSize: number,
    public learningRate: number = 0.05
  ) {
    this.weightsInputHidden = Array.from({ length: inputSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * Math.sqrt(2 / inputSize))
    );
    this.weightsHiddenOutput = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: outputSize }, () => (Math.random() - 0.5) * Math.sqrt(2 / hiddenSize))
    );
    this.biasHidden = new Array(hiddenSize).fill(0);
    this.biasOutput = new Array(outputSize).fill(0);
  }

  public relu = (x: number) => Math.max(0, x);
  public reluDerivative = (x: number) => (x > 0 ? 1 : 0);

  public softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const exps = logits.map((val) => Math.exp(val - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map((val) => val / (sumExps || 1));
  }

  public trainStep(inputVector: number[], target: number[]): number {
    const hiddenOutputs = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.biasHidden[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputVector[i] * this.weightsInputHidden[i][j];
      }
      hiddenOutputs[j] = this.relu(sum);
    }

    const logits = new Array(this.outputSize).fill(0);
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.biasOutput[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenOutputs[j] * this.weightsHiddenOutput[j][k];
      }
      logits[k] = sum;
    }

    const finalOutputs = this.softmax(logits);
    const outputErrors = target.map((t, k) => t - finalOutputs[k]);

    const hiddenErrors = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let error = 0;
      for (let k = 0; k < this.outputSize; k++) {
        error += outputErrors[k] * this.weightsHiddenOutput[j][k];
      }
      hiddenErrors[j] = error * this.reluDerivative(hiddenOutputs[j]);
    }

    for (let k = 0; k < this.outputSize; k++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.weightsHiddenOutput[j][k] += this.learningRate * outputErrors[k] * hiddenOutputs[j];
      }
      this.biasOutput[k] += this.learningRate * outputErrors[k];
    }
    for (let j = 0; j < this.hiddenSize; j++) {
      for (let i = 0; i < this.inputSize; i++) {
        this.weightsInputHidden[i][j] += this.learningRate * hiddenErrors[j] * inputVector[i];
      }
      this.biasHidden[j] += this.learningRate * hiddenErrors[j];
    }
    return outputErrors.reduce((a, b) => a + b * b, 0);
  }
  
  public predict(inputVector: number[]): number[] {
    // 1. Hidden Layer
    const hiddenOutputs = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.biasHidden[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputVector[i] * this.weightsInputHidden[i][j];
      }
      hiddenOutputs[j] = this.relu(sum);
    }

    // 2. Output Layer
    const logits = new Array(this.outputSize).fill(0);
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.biasOutput[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenOutputs[j] * this.weightsHiddenOutput[j][k];
      }
      logits[k] = sum;
    }

    return this.softmax(logits);
  }

  // 1. Tambahin Decoder (untuk ngebentuk output dari Latent Space)
  // Input: latentVector (size kecil), Output: generatedData (size inputSize)
  public generate(latentVector: number[]): number[] {
    const hidden = new Array(this.hiddenSize).fill(0);
    // Forward pass dari latent ke hidden
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.biasHidden[j]; // Pake bias yang ada
      for (let i = 0; i < latentVector.length; i++) {
        sum += latentVector[i] * this.weightsInputHidden[i][j];
      }
      hidden[j] = this.relu(sum);
    }
    
    // Output layer (ngebentuk data original)
    const output = new Array(this.inputSize).fill(0);
    for (let k = 0; k < this.inputSize; k++) {
      let sum = this.biasOutput[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hidden[j] * this.weightsHiddenOutput[j][k];
      }
      // Pake sigmoid di sini biar outputnya 0-1 (normalisasi data)
      output[k] = 1 / (1 + Math.exp(-sum)); 
    }
    return output;
  }

// Tambahin method ini di class NeuralNetwork lo
public reparameterize(mu: number[], logVar: number[]): number[] {
  return mu.map((m, i) => {
    const std = Math.exp(0.5 * logVar[i]); // std = e^(0.5 * logVar)
    const epsilon = this.sampleNormal();   // Angka acak dari distribusi N(0,1)
    return m + (std * epsilon);
  });
}

// Helper buat dapet noise distribusi normal (Box-Muller transform)
private sampleNormal(): number {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

  // Tambahin ini biar save/load tetep jalan
  public getModel(): any {
    return {
      w1: this.weightsInputHidden,
      w2: this.weightsHiddenOutput,
      b1: this.biasHidden,
      b2: this.biasOutput
    };
  }

  public loadModel(m: any): void {
    this.weightsInputHidden = m.w1;
    this.weightsHiddenOutput = m.w2;
    this.biasHidden = m.b1;
    this.biasOutput = m.b2;
  }
}
