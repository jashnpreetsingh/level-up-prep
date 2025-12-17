import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const db = new Database(path.join(__dirname, 'shadow_monarch.db'));

// Initialize database tables
db.exec(`
  CREATE TABLE IF NOT EXISTS user_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    str INTEGER DEFAULT 10,
    int INTEGER DEFAULT 10,
    spd INTEGER DEFAULT 10,
    xp INTEGER DEFAULT 0,
    legions TEXT DEFAULT '{"Infantry":0,"Mage":0,"Assassin":0,"Tank":0,"Commander":0,"Construct":0,"Rune":0}',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS completed_problems (
    id TEXT PRIMARY KEY,
    context TEXT NOT NULL,
    completed INTEGER DEFAULT 1,
    retreat INTEGER DEFAULT 0,
    notes TEXT,
    timestamp INTEGER,
    date_solved TEXT,
    time_complexity TEXT,
    space_complexity TEXT,
    time_taken INTEGER,
    last_reviewed INTEGER,
    review_count INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS unlocked_shadows (
    name TEXT PRIMARY KEY,
    rank TEXT,
    origin TEXT
  );

  CREATE TABLE IF NOT EXISTS save_slots (
    slot INTEGER PRIMARY KEY,
    date TEXT,
    data TEXT
  );

  CREATE TABLE IF NOT EXISTS daily_activity (
    date TEXT PRIMARY KEY,
    problems INTEGER DEFAULT 0,
    time_spent INTEGER DEFAULT 0
  );

  -- Initialize default user state if not exists
  INSERT OR IGNORE INTO user_state (id) VALUES (1);
`);

// API functions
export function getState() {
  const userState = db.prepare('SELECT * FROM user_state WHERE id = 1').get();
  const completed = db.prepare('SELECT * FROM completed_problems').all();
  const shadows = db.prepare('SELECT * FROM unlocked_shadows').all();
  const saves = db.prepare('SELECT * FROM save_slots ORDER BY slot').all();
  const dailyActivity = db.prepare('SELECT * FROM daily_activity').all();

  const completedMap = {};
  completed.forEach(c => {
    completedMap[c.id] = {
      completed: Boolean(c.completed),
      retreat: Boolean(c.retreat),
      notes: c.notes,
      timestamp: c.timestamp,
      dateSolved: c.date_solved,
      timeComplexity: c.time_complexity,
      spaceComplexity: c.space_complexity,
      context: c.context,
      timeTaken: c.time_taken,
      lastReviewed: c.last_reviewed,
      reviewCount: c.review_count
    };
  });

  const saveSlotsArray = [null, null, null];
  saves.forEach(s => {
    if (s.slot >= 0 && s.slot < 3) {
      saveSlotsArray[s.slot] = { date: s.date, data: s.data };
    }
  });

  const dailyActivityMap = {};
  dailyActivity.forEach(d => {
    dailyActivityMap[d.date] = { problems: d.problems, time: d.time_spent };
  });

  return {
    stats: {
      str: userState.str,
      int: userState.int,
      spd: userState.spd
    },
    xp: userState.xp,
    legions: JSON.parse(userState.legions),
    completed: completedMap,
    unlockedShadows: shadows,
    saveSlots: saveSlotsArray,
    dailyActivity: dailyActivityMap
  };
}

export function saveState(state) {
  const updateStats = db.prepare(`
    UPDATE user_state 
    SET str = ?, int = ?, spd = ?, xp = ?, legions = ?, updated_at = CURRENT_TIMESTAMP
    WHERE id = 1
  `);
  
  updateStats.run(
    state.stats.str,
    state.stats.int,
    state.stats.spd,
    state.xp,
    JSON.stringify(state.legions)
  );

  // Clear and re-insert completed problems
  if (state.completed) {
    const upsertCompleted = db.prepare(`
      INSERT OR REPLACE INTO completed_problems 
      (id, context, completed, retreat, notes, timestamp, date_solved, time_complexity, space_complexity, time_taken, last_reviewed, review_count)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    Object.entries(state.completed).forEach(([id, data]) => {
      upsertCompleted.run(
        id,
        data.context || '',
        data.completed ? 1 : 0,
        data.retreat ? 1 : 0,
        data.notes || '',
        data.timestamp || null,
        data.dateSolved || '',
        data.timeComplexity || '',
        data.spaceComplexity || '',
        data.timeTaken || 0,
        data.lastReviewed || null,
        data.reviewCount || 0
      );
    });
  }

  // Update shadows
  if (state.unlockedShadows) {
    const upsertShadow = db.prepare(`
      INSERT OR REPLACE INTO unlocked_shadows (name, rank, origin) VALUES (?, ?, ?)
    `);
    state.unlockedShadows.forEach(s => {
      upsertShadow.run(s.name, s.rank, s.origin);
    });
  }

  // Update save slots
  if (state.saveSlots) {
    const upsertSave = db.prepare(`
      INSERT OR REPLACE INTO save_slots (slot, date, data) VALUES (?, ?, ?)
    `);
    state.saveSlots.forEach((save, idx) => {
      if (save) {
        upsertSave.run(idx, save.date, save.data);
      }
    });
  }

  return { success: true };
}

export function completeProblem(id, data) {
  const stmt = db.prepare(`
    INSERT OR REPLACE INTO completed_problems 
    (id, context, completed, retreat, notes, timestamp, date_solved, time_complexity, space_complexity, time_taken, last_reviewed, review_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  stmt.run(
    id,
    data.context || '',
    1,
    data.retreat ? 1 : 0,
    data.notes || '',
    data.timestamp || Date.now(),
    data.dateSolved || new Date().toLocaleDateString(),
    data.timeComplexity || '',
    data.spaceComplexity || '',
    data.timeTaken || 0,
    data.lastReviewed || Date.now(),
    data.reviewCount || 0
  );

  return { success: true };
}

export function markReviewed(id) {
  const current = db.prepare('SELECT review_count FROM completed_problems WHERE id = ?').get(id);
  if (current) {
    db.prepare('UPDATE completed_problems SET review_count = ?, last_reviewed = ? WHERE id = ?')
      .run(current.review_count + 1, Date.now(), id);
  }
  return { success: true };
}

export function saveSlot(slot, data) {
  db.prepare('INSERT OR REPLACE INTO save_slots (slot, date, data) VALUES (?, ?, ?)')
    .run(slot, new Date().toLocaleString(), data);
  return { success: true };
}

export function loadSlot(slot) {
  const save = db.prepare('SELECT * FROM save_slots WHERE slot = ?').get(slot);
  return save || null;
}

export function resetAll() {
  db.exec(`
    DELETE FROM completed_problems;
    DELETE FROM unlocked_shadows;
    DELETE FROM save_slots;
    DELETE FROM daily_activity;
    UPDATE user_state SET str = 10, int = 10, spd = 10, xp = 0, 
      legions = '{"Infantry":0,"Mage":0,"Assassin":0,"Tank":0,"Commander":0,"Construct":0,"Rune":0}'
      WHERE id = 1;
  `);
  return { success: true };
}

export default db;
