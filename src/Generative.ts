import { NeuralNetwork } from './NeuralNetwork.js';

export class Generative extends NeuralNetwork {
  // Encoder weights: buat memetakan input ke mean (mu) dan log-variance (logVar)
  private weightsEncoder: number[][];

  constructor(
    public inputSize: number,
    public hiddenSize: number,
    public latentSize: number, // Ukuran "Bottle Neck" / Latent Space
    public learningRate: number = 0.05
  ) {
    // Panggil constructor NeuralNetwork (Output layer kita set ke inputSize buat rekonstruksi)
    super(inputSize, hiddenSize, inputSize, learningRate);
    
    // Inisialisasi Encoder untuk mencari mu & logVar
    this.weightsEncoder = Array.from({ length: inputSize }, () =>
      Array.from({ length: latentSize * 2 }, () => (Math.random() - 0.5) * Math.sqrt(2 / inputSize))
    );
  }

public encode(inputVector: number[]): { mu: number[], logVar: number[] } {
  const encoderOutput = new Array(this.latentSize * 2).fill(0);
  for (let j = 0; j < this.latentSize * 2; j++) {
    let sum = 0;
    for (let i = 0; i < this.inputSize; i++) {
      sum += inputVector[i] * this.weightsEncoder[i][j];
    }
    // WAJIB: Pake aktivasi di encoder biar non-linear
    encoderOutput[j] = this.relu(sum); 
  }
  return {
    mu: encoderOutput.slice(0, this.latentSize),
    logVar: encoderOutput.slice(this.latentSize)
  };
}

public trainVAE(inputVector: number[]): number {
  const { mu, logVar } = this.encode(inputVector);
  const z = this.reparameterize(mu, logVar);
  const reconstruction = this.generate(z);

  // 1. Hitung Error
  let reconLoss = 0;
  const encoderGradients = Array.from({ length: this.inputSize }, () => new Array(this.latentSize * 2).fill(0));

  for (let i = 0; i < this.inputSize; i++) {
    const val = Math.max(0.001, Math.min(0.999, reconstruction[i]));
    reconLoss -= (inputVector[i] * Math.log(val) + (1 - inputVector[i]) * Math.log(1 - val));
    
    // Update sederhana: dorong weightsEncoder ke arah error
    const delta = (reconstruction[i] - inputVector[i]) * this.learningRate;
    for(let j = 0; j < this.latentSize * 2; j++) {
      this.weightsEncoder[i][j] -= delta * 0.1; // *Learning Rate kecil buat encoder
    }
  }

  // 2. KL Divergence (biar mu mendekati 0 dan logVar mendekati 0)
  const beta = 0.01; 
  const klDivergence = -0.5 * beta * mu.reduce((acc, m, i) => 
    acc + (1 + logVar[i] - Math.pow(m, 2) - Math.exp(logVar[i])), 0);

  return reconLoss + klDivergence;
}

public predict(inputVector: number[]): number[] {
  // 1. Encode: Masukin data ke "Bottle Neck" biar dapet mu dan logVar
  const { mu, logVar } = this.encode(inputVector);
  
  // 2. Reparameterize: Kita ambil nilai mean (mu) aja 
  // (Pas prediksi/inference, kita gak butuh noise epsilon, biar hasilnya stabil)
  const z = mu; 
  
  // 3. Generate: Bangun ulang data dari latent vector (z)
  return this.generate(z);
}

}
