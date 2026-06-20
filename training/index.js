import { Generative } from '../dist/index.min.mjs';

// 1. Kamus data yang konsisten
const vocabulary = ["fix", "add", "bug", "style", "refactor", "docs", "feature", "test", "chore", "perf", "code"];
const inputSize = vocabulary.length;

const textToVector = (text) => vocabulary.map(word => text.split(" ").includes(word) ? 1 : 0);

// Threshold dibuat lebih ketat biar AI cuma nampilin kata yang "yakin"
const vectorToText = (vector) => {
  return vector.map((val, i) => val > 0.6 ? vocabulary[i] : "").filter(Boolean).join(" ");
};

// 2. Dataset yang proper (Pola-pola yang emang beneran ada di real-world commit)
const trainingData = [
  "fix bug", "fix perf", "fix refactor", "fix bug perf",
  "add feature", "add test", "add feature perf",
  "refactor style", "refactor code", "refactor style code",
  "docs test", "docs refactor", "chore docs"
];

// 3. Setup Otak
const brain = new Generative(inputSize, 32, 6, 0.001); 

console.log("Mulai training dengan dataset terstruktur...");

// 4. Training Loop yang proper (dengan Shuffling)
for (let epoch = 0; epoch < 5000; epoch++) {
  let totalLoss = 0;
  
  // Shuffle data tiap epoch biar model nggak ngafalin urutan
  const shuffledData = [...trainingData].sort(() => Math.random() - 0.5);
  
  for (const sentence of shuffledData) {
    totalLoss += brain.trainVAE(textToVector(sentence));
  }
  
  if (epoch % 1000 === 0) {
    console.log(`Epoch ${epoch} | Avg Loss: ${(totalLoss / trainingData.length).toFixed(4)}`);
  }
}

// 5. Test Predict
const testInput = textToVector("fix");
const resultVector = brain.predict(testInput);

console.log("----------------------------");
console.log("Input: fix");
console.log("Prediksi AI:", vectorToText(resultVector));
