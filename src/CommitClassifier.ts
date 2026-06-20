/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { NeuralNetwork } from './NeuralNetwork.js';
import * as fs from 'fs';

export const CATEGORIES = [
  'FEATURES',
  'BUG_FIXES',
  'MAINTENANCE',
  'UNCATEGORIZED',
] as const;
type Category = (typeof CATEGORIES)[number];

export class CommitClassifier {
  private brain: NeuralNetwork | null = null;
  private vocabulary: string[] = [];
  private hiddenSize = 8;
  private learningRate = 0.05;
  private confidenceThreshold = 0.55;
  private activation: 'relu' | 'swish';
  private idfWeights: number[] = [];
  private datasetSize: number = 0;

  constructor(
    hiddenSize: number = 8,
    learningRate: number = 0.05,
    confidenceThreshold: number = 0.55,
    activation: 'relu' | 'swish' = 'relu',
  ) {
    this.hiddenSize = hiddenSize;
    this.learningRate = learningRate;
    this.confidenceThreshold = confidenceThreshold;
    this.activation = activation;
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

  public train(
    data: { text: string; category: Category }[],
    epochs = 200,
  ): void {
    this.buildVocabulary(data.map((d) => d.text));

    this.brain = new NeuralNetwork(
      this.vocabulary.length,
      this.hiddenSize,
      CATEGORIES.length,
      this.learningRate,
    );

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochError = 0;
      for (const item of data) {
        const inputVector = this.textToVector(item.text);
        const target = new Array(CATEGORIES.length).fill(0);
        target[CATEGORIES.indexOf(item.category)] = 1;

        epochError += this.brain.trainStep(inputVector, target);
      }
      this.drawNeuralDashboard(epoch, epochs, epochError / data.length);
    }
    console.log('\n✅ Training selesai!');
  }

  public getCommitQualityScore(text: string): number {
    const inputVector = this.textToVector(text);

    const tokens = this.tokenizeToNgrams(text);
    if (tokens.length < 2) return 0;

    const probs = this.brain!.predict(inputVector);

    const entropy = -probs.reduce(
      (acc, p) => acc + (p > 0 ? p * Math.log2(p) : 0),
      0,
    );

    const normalizedEntropy = entropy / Math.log2(CATEGORIES.length);

    return 1 - normalizedEntropy;
  }

  public classify(text: string): Category {
    if (!this.brain || this.vocabulary.length === 0) return 'MAINTENANCE';

    const qualityScore = this.getCommitQualityScore(text);
    const minQualityThreshold = 0.25;

    if (qualityScore < minQualityThreshold) {
      return 'UNCATEGORIZED';
    }

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

  private buildVocabulary(dataset: string[]): void {
    this.datasetSize = dataset.length;
    const vocabSet = new Set<string>();
    const docCount: Record<string, number> = {};

    dataset.forEach((text) => {
      const tokens = new Set(this.tokenizeToNgrams(text)); // Pake Set biar 1 per doc
      tokens.forEach((token) => {
        vocabSet.add(token);
        docCount[token] = (docCount[token] || 0) + 1;
      });
    });

    this.vocabulary = Array.from(vocabSet);
    
    // Hitung IDF untuk setiap kata di vocab
    this.idfWeights = this.vocabulary.map((token) => {
      return Math.log(this.datasetSize / (docCount[token] || 1)) + 1;
    });
  }

  // 3. Update textToVector jadi TF-IDF
  private textToVector(text: string): number[] {
    const vector = new Array(this.vocabulary.length).fill(0);
    const tokens = this.tokenizeToNgrams(text);
    
    // Hitung TF (Term Frequency)
    const tfMap: Record<string, number> = {};
    tokens.forEach((t) => (tfMap[t] = (tfMap[t] || 0) + 1));

    // TF * IDF
    this.vocabulary.forEach((token, idx) => {
      if (tfMap[token]) {
        const tf = tfMap[token] / tokens.length;
        vector[idx] = tf * this.idfWeights[idx];
      }
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

    const localVocabSet = new Set<string>();
    commits.forEach((commit) => {
      this.tokenizeToNgrams(commit).forEach((token) =>
        localVocabSet.add(token),
      );
    });
    const localVocabulary = Array.from(localVocabSet);

    const docFrequency: Record<string, number> = {};
    localVocabulary.forEach((token) => (docFrequency[token] = 0));

    commits.forEach((commit) => {
      const tokens = new Set(this.tokenizeToNgrams(commit));
      tokens.forEach((token) => {
        if (docFrequency[token] !== undefined) docFrequency[token]++;
      });
    });

    const textToTFIDFVector = (text: string): number[] => {
      const vector = new Array(localVocabulary.length).fill(0);
      const tokens = this.tokenizeToNgrams(text);

      const tfMap: Record<string, number> = {};
      tokens.forEach((t) => (tfMap[t] = (tfMap[t] || 0) + 1));

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

  private drawNeuralDashboard(
    epoch: number,
    totalEpochs: number,
    error: number,
  ): void {
    process.stdout.cursorTo(0, 0);

    const percent = Math.floor((epoch / totalEpochs) * 100);
    const bar =
      '█'.repeat(Math.floor(percent / 5)) +
      '-'.repeat(20 - Math.floor(percent / 5));
    process.stdout.write(
      `[${bar}] ${percent}% | Eror: ${error.toFixed(6)}\n\n`,
    );

    // GANTI DI SINI: Akses ke l1.weights yang ada di dalam brain
    const weights = this.brain ? this.brain['l1'].weights : [];
    const rows = Math.min(weights.length, 6);

    process.stdout.write(
      '      I N P U T  -->  H I D D E N  -->  O U T P U T\n',
    );

    for (let i = 0; i < rows; i++) {
      let rowStr = `Node ${i} `;

      // Pastikan weights[i] ada isinya
      if (weights[i]) {
        for (let j = 0; j < Math.min(weights[i].length, 3); j++) {
          const weight = weights[i][j] || 0;
          const color = weight > 0 ? '\x1b[32m' : '\x1b[31m';
          const synapse = (i + j) % 2 === 0 ? '◢' : '◣';
          rowStr += ` ${color}${synapse}\x1b[0m `;
        }
      }

      rowStr += ' ● ● ●';
      process.stdout.write(rowStr + '\n');
    }
  }

public saveModel(filePath: string): void {
  if (!this.brain) return;
  const modelState = {
    vocabulary: this.vocabulary,
    idfWeights: this.idfWeights, // <--- TAMBAH INI
    brain: this.brain.getModel(),
  };
  fs.writeFileSync(filePath, JSON.stringify(modelState), 'utf8'); // Hapus null, 2 biar file lebih ringan
}

public loadModel(filePath: string): boolean {
  if (!fs.existsSync(filePath)) return false;
  const modelState = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  
  this.vocabulary = modelState.vocabulary;
  this.idfWeights = modelState.idfWeights; // <--- TAMBAH INI
  
  this.brain = new NeuralNetwork(
    this.vocabulary.length,
    this.hiddenSize,
    CATEGORIES.length,
    this.learningRate, // Pastikan learningRate ada kalau dibutuhkan di NeuralNetwork
  );
  this.brain.loadModel(modelState.brain);
  return true;
}

}
