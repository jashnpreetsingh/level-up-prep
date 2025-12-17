// Script to recalculate stats based on completed problems
// Run this with: node scripts/recalculate-stats.js

import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const db = new Database(path.join(__dirname, '..', 'server', 'shadow_monarch.db'));

// Get all completed problems
const completed = db.prepare('SELECT * FROM completed_problems').all();

console.log(`Found ${completed.length} completed problems`);

let str = 10, int = 10, spd = 10, xp = 0;

completed.forEach(problem => {
    let strGain = 0, intGain = 0, spdGain = 0;

    const context = problem.context;
    const retreat = problem.retreat;
    const timeTaken = problem.time_taken || 20;

    // Default time limit based on context
    const limit = context === 'DSA' ? 20 : 60;

    if (context === 'DSA') {
        // STR gains based on difficulty (guessing from problem name patterns)
        strGain = 1;

        // SPD based on time taken
        if (timeTaken <= limit / 2) spdGain = 3;
        else if (timeTaken <= limit) spdGain = 1;
        else spdGain = -1;

        // DSA should also give some INT (was missing!)
        intGain = 1;

    } else if (context === 'BEHAVIORAL') {
        // Behavioral gives INT
        intGain = 3; // Assuming Detailed average

    } else {
        // ML/MATH gives INT + STR
        intGain = 3; // Assuming average understanding
        strGain = 2;
    }

    // Tactical retreat bonus
    if (retreat) {
        intGain += 3;
        spdGain = 0;
    }

    str += strGain;
    int += intGain;
    spd = Math.max(0, spd + spdGain);
    xp += (strGain + intGain + Math.max(0, spdGain)) * 10;

    console.log(`  ${problem.id}: STR+${strGain}, INT+${intGain}, SPD${spdGain >= 0 ? '+' : ''}${spdGain}`);
});

console.log('\n--- RECALCULATED STATS ---');
console.log(`STR: ${str}`);
console.log(`INT: ${int}`);
console.log(`SPD: ${spd}`);
console.log(`XP: ${xp}`);

// Update the database
db.prepare('UPDATE user_state SET str = ?, int = ?, spd = ?, xp = ? WHERE id = 1')
    .run(str, int, spd, xp);

console.log('\n✓ Stats updated in database!');

db.close();
