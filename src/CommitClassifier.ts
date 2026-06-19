import { NeuralNetwork } from './NeuralNetwork.js';
import * as fs from 'fs';

// Tambahin kategori baru
export const CATEGORIES = ['FEATURES', 'BUG_FIXES', 'MAINTENANCE', 'UNCATEGORIZED'] as const;
type Category = (typeof CATEGORIES)[number];

export class CommitClassifier {
  private brain: NeuralNetwork | null = null;
  private vocabulary: string[] = [];
  private hiddenSize = 8;
  private learningRate = 0.05;
  private confidenceThreshold = 0.55;

  constructor(
    hiddenSize: number = 8,
    learningRate: number = 0.05,
    confidenceThreshold: number = 0.55
  ) {
    this.hiddenSize = hiddenSize;
    this.learningRate = learningRate;
    this.confidenceThreshold = confidenceThreshold;
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((word) => word.length > 1);
  }

  private tokenizeToNgrams(text: string): string[] {
    const words = this.tokenize(text);
    const ngrams = [...words];
    
    for (let i = 0; i < words.length - 1; i++) {
      ngrams.push(`${words[i]}_${words[i + 1]}`);
    }
    return ngrams;
  }

  public train(data: { text: string; category: Category }[], epochs = 200): void {
    this.buildVocabulary(data.map(d => d.text));
    
    // Inisialisasi si Otak di sini
    this.brain = new NeuralNetwork(this.vocabulary.length, this.hiddenSize, CATEGORIES.length, this.learningRate);

    console.log(`🧠 Training NN dengan ${this.vocabulary.length} fitur...`);

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochError = 0;
      for (const item of data) {
        const inputVector = this.textToVector(item.text);
        const target = new Array(CATEGORIES.length).fill(0);
        target[CATEGORIES.indexOf(item.category)] = 1;
        
        // Panggil fungsi "Otak"
        epochError += this.brain.trainStep(inputVector, target);
      }
      this.drawNeuralDashboard(epoch, epochs, epochError / data.length);
    }
    console.log('\n✅ Training selesai!');
  }

  public classify(text: string): Category {
    if (!this.brain || this.vocabulary.length === 0) return 'MAINTENANCE';

    const inputVector = this.textToVector(text);
    const finalOutputs = this.brain.predict(inputVector);

    let maxIdx = 0;
    let maxVal = -1;
    for (let k = 0; k < finalOutputs.length; k++) {
      if (finalOutputs[k] > maxVal) {
        maxVal = finalOutputs[k];
        maxIdx = k;
      }
    }
    
    if (maxVal < this.confidenceThreshold) {
      return 'UNCATEGORIZED';
    }
    
    return CATEGORIES[maxIdx];
  }

  // 2. Build Vocabulary menggunakan N-gram dari dataset training yang sama
  private buildVocabulary(dataset: string[]): void {
    const vocabSet = new Set<string>();
    dataset.forEach((text) => {
      this.tokenizeToNgrams(text).forEach((token) => vocabSet.add(token));
    });
    this.vocabulary = Array.from(vocabSet);
  }

  // 3. Ubah teks ke representasi vektor berdasarkan N-gram Vocabulary
  private textToVector(text: string): number[] {
    const vector = new Array(this.vocabulary.length).fill(0);
    const tokens = this.tokenizeToNgrams(text);
    tokens.forEach((token) => {
      const idx = this.vocabulary.indexOf(token);
      if (idx !== -1) vector[idx] += 1;
    });
    return vector;
  }

  private calculateCosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0; 
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  public generateChangelog(commits: string[]): string {
    const groups: Record<Category, string[]> = {
      FEATURES: [],
      BUG_FIXES: [],
      MAINTENANCE: [],
      UNCATEGORIZED: [],
    };
    
    // Bangun Vocabulary Lokal khusus untuk lingkup commit saat ini
    const localVocabSet = new Set<string>();
    commits.forEach((commit) => {
      this.tokenizeToNgrams(commit).forEach((token) => localVocabSet.add(token));
    });
    const localVocabulary = Array.from(localVocabSet);
    
    // Hitung Document Frequency (DF) untuk formula TF-IDF lokal
    const docFrequency: Record<string, number> = {};
    localVocabulary.forEach(token => docFrequency[token] = 0);
    
    commits.forEach(commit => {
      const tokens = new Set(this.tokenizeToNgrams(commit));
      tokens.forEach(token => {
        if (docFrequency[token] !== undefined) docFrequency[token]++;
      });
    });

    // Transformasi teks commit berjalan ke vektor berbasis TF-IDF murni
    const textToTFIDFVector = (text: string): number[] => {
      const vector = new Array(localVocabulary.length).fill(0);
      const tokens = this.tokenizeToNgrams(text);
      
      const tfMap: Record<string, number> = {};
      tokens.forEach(t => tfMap[t] = (tfMap[t] || 0) + 1);
      
      localVocabulary.forEach((token, idx) => {
        if (tfMap[token]) {
          const tf = tfMap[token] / tokens.length;
          const idf = Math.log(commits.length / (docFrequency[token] || 1)) + 1;
          vector[idx] = tf * idf;
        }
      });
      return vector;
    };

    const processedVectors: number[][] = [];
    // Threshold diturunkan ke 0.75 karena seleksi TF-IDF N-gram jauh lebih ketat & sensitif
    const similarityThreshold = 0.75; 

    commits.forEach((commit) => {
      const currentVector = textToTFIDFVector(commit);

      let isDuplicateByAI = false;
      for (const existingVector of processedVectors) {
        const similarity = this.calculateCosineSimilarity(
          currentVector,
          existingVector,
        );

        if (similarity > similarityThreshold) {
          isDuplicateByAI = true;
          break;
        }
      }

      if (!isDuplicateByAI) {
        processedVectors.push(currentVector);
        
        // Klasifikasi tetap berjalan aman karena textToVector() di atas sudah otomatis dukung N-gram
        const category = this.classify(commit);
        groups[category].push(commit);
      }
    });

    let markdown = '';
    if (groups.FEATURES.length > 0)
      markdown += `### 🚀 Features\n${groups.FEATURES.join('\n')}\n\n`;
    if (groups.BUG_FIXES.length > 0)
      markdown += `### 🐛 Bug Fixes\n${groups.BUG_FIXES.join('\n')}\n\n`;
    if (groups.MAINTENANCE.length > 0)
      markdown += `### 🧹 Maintenance\n${groups.MAINTENANCE.join('\n')}\n\n`;

    return markdown;
  }

  private drawNeuralDashboard(epoch: number, totalEpochs: number, error: number): void {
    process.stdout.cursorTo(0, 0);

    const percent = Math.floor((epoch / totalEpochs) * 100);
    const bar = '█'.repeat(Math.floor(percent / 5)) + '-'.repeat(20 - Math.floor(percent / 5));
    process.stdout.write(`[${bar}] ${percent}% | Eror: ${error.toFixed(6)}\n\n`);

    // Kita bikin grid "Mesh" biar gak datar
    // Kita ambil sampel dari weightsInputHidden buat representasi visual
    const weights = this.brain?.weightsInputHidden || [];
    const rows = Math.min(weights.length, 6); // Batasi biar terminal gak scroll

    process.stdout.write("      I N P U T  -->  H I D D E N  -->  O U T P U T\n");

    for (let i = 0; i < rows; i++) {
      let rowStr = `Node ${i} `;
      
      // Efek Jaring (Mesh): looping buat bikin visualisasi garis silang
      for (let j = 0; j < 3; j++) { 
        const weight = weights[i][j] || 0;
        const color = weight > 0 ? "\x1b[32m" : "\x1b[31m";
        
        // Simbol yang bikin efek diagonal/jaring
        const synapse = (i + j) % 2 === 0 ? "◢" : "◣"; 
        rowStr += ` ${color}${synapse}\x1b[0m `;
      }
      
      rowStr += " ● ● ●"; // Hidden nodes
      process.stdout.write(rowStr + "\n");
    }
  }


  public saveModel(filePath: string): void {
    if (!this.brain) return;
    const modelState = { vocabulary: this.vocabulary, brain: this.brain.getModel() };
    fs.writeFileSync(filePath, JSON.stringify(modelState, null, 2), 'utf8');
  }

  public loadModel(filePath: string): boolean {
    if (!fs.existsSync(filePath)) return false;
    const modelState = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    this.vocabulary = modelState.vocabulary;
    this.brain = new NeuralNetwork(this.vocabulary.length, this.hiddenSize, CATEGORIES.length);
    this.brain.loadModel(modelState.brain);
    return true;
  }
}
