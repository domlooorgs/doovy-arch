import { CommitClassifier } from '../dist/index.min.mjs';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import * as path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const classifier = new CommitClassifier(64, 0.02, 0.75, 'swish');

process.on('SIGINT', () => {
  console.log('\n\n⚠️ CTRL+C terdeteksi! Sedang menyimpan progress...');
  classifier.saveModel('checkpoint-model.json');
  console.log('✅ Progress tersimpan di checkpoint-model.json. Exiting...');
  process.exit();
});

// Fungsi loadDataset yang udah di-update sesuai acuan data lo
const loadDataset = () => {
  const files = [
    { file: 'datasets/features.json', cat: 'FEATURES' },
    { file: 'datasets/bug_fixed.json', cat: 'BUG_FIXES' },
    { file: 'datasets/maintenance.json', cat: 'MAINTENANCE' },
    { file: 'datasets/junk.json', cat: 'UNCATEGORIZED' }
  ];

  let dataset = [];

  files.forEach(({ file, cat }) => {
    const filePath = path.resolve(__dirname, file);
    
    try {
      if (fs.existsSync(filePath)) {
        const raw = fs.readFileSync(filePath, 'utf8');
        const content = JSON.parse(raw);

        const formatted = content.map((item, index) => {
          // DEBUG: Cek tipe data tiap item
          if (!item) return null;
          
          let result;
          if (typeof item === 'string') {
            result = { text: item, category: cat };
          } else {
            result = { text: item.text, category: item.category || cat };
          }
          
          if (!result.text) console.warn(`⚠️ Warning: Item ke-${index} di ${file} tidak punya text!`);
          return result;
        }).filter(item => item !== null);

        dataset = [...dataset, ...formatted];
      } else {
        console.warn(`⚠️ File tidak ditemukan: ${filePath}`);
      }
    } catch (e) {
      console.error(`❌ Gagal membaca ${file}:`, e);
    }
  });

  return dataset;
};

const data = loadDataset();

console.log("🚀 Starting Training...");
// Epoh dinaikin biar konvergen
classifier.train(data, 300);

// Test Klasifikasi
const tests = [
  "fix: button alignment issue",
  "feat: add export to pdf",
  "chore: bump version",
  "fix subagia kena bug turunin feature chore monyet"
];

console.log("\n🧪 Hasil Testing:");
tests.forEach(t => {
  console.log(`Input: "${t}" -> Hasil: ${classifier.classify(t)}`);
});

// Test Changelog
const changelog = classifier.generateChangelog([
  "feat: add dark mode",
  "fix: login button alignment",
  "feat: add dark mode", // Harusnya ke-filter sebagai duplikat
  "chore: clean temp files",
  "fix: resolve infinite loop in fetch",
  "fix subagia kena bug turunin feature chore monyet",
  "fix: fix fix fix",
  "finally make this thing stop exploding",
  "remove workaround because workaround for workaround no longer needed",
  "please work",
  "capek gua ngentot!"
]);

console.log("\n📄 Changelog Preview:");
console.log(changelog);