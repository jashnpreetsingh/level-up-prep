import { useState, useEffect, useRef } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import Header from './components/Header';
import QuestContainer from './components/QuestContainer';
import HuntModal from './components/HuntModal';
import ReportModal from './components/ReportModal';
import Sidebar from './components/Sidebar';
import StudyModal from './components/StudyModal';
import * as api from './hooks/useApi';
import { generateId } from './data/problemData';
import { getTopicContent } from './data/content';
import './index.css';

const DEFAULT_STATE = {
  stats: { str: 10, int: 10, spd: 10 },
  xp: 0,
  legions: { Infantry: 0, Mage: 0, Assassin: 0, Tank: 0, Commander: 0, Construct: 0, Rune: 0 },
  completed: {},
  unlockedShadows: [],
  saveSlots: [null, null, null],
  dailyActivity: {}
};

function App() {
  const [state, setState] = useState(DEFAULT_STATE);
  const [loading, setLoading] = useState(true);
  const [currentContext, setCurrentContext] = useState('DSA');
  const [activeHunt, setActiveHunt] = useState(null);
  const [huntResult, setHuntResult] = useState(null);
  const [showSaveIndicator, setShowSaveIndicator] = useState(false);
  const [countdown, setCountdown] = useState('LOADING...');
  const [studyTopic, setStudyTopic] = useState(null);
  const fileInputRef = useRef(null);

  // Load state from backend
  useEffect(() => {
    loadState();
    startCountdown();
  }, []);

  const loadState = async () => {
    try {
      const data = await api.fetchState();
      setState(prev => ({ ...DEFAULT_STATE, ...data }));
    } catch (err) {
      console.error('Failed to load state:', err);
      // Use default state if backend fails
    } finally {
      setLoading(false);
    }
  };

  const saveStateToDB = async (newState) => {
    try {
      await api.saveState(newState);
      setShowSaveIndicator(true);
      setTimeout(() => setShowSaveIndicator(false), 2000);
    } catch (err) {
      console.error('Failed to save:', err);
    }
  };

  const startCountdown = () => {
    const updateTimer = () => {
      const target = new Date('2026-01-05T09:00:00').getTime();
      const now = Date.now();
      const diff = target - now;

      if (diff < 0) {
        setCountdown('GATE OPEN');
        return;
      }

      const d = Math.floor(diff / (1000 * 60 * 60 * 24));
      const h = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const m = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      setCountdown(`${d}D ${h}H ${m}M`);
    };

    updateTimer();
    setInterval(updateTimer, 60000);
  };

  const handleStartHunt = (title, context, meta) => {
    setActiveHunt({ title, context, meta });
  };

  const handleHuntComplete = (result) => {
    setActiveHunt(null);
    setHuntResult(result);
  };

  const handleHuntCancel = () => {
    setActiveHunt(null);
  };

  const handleReportSubmit = async (report) => {
    const pid = generateId(report.context, report.title);

    // Calculate stat gains
    let strGain = 1, intGain = 0, spdGain = 0;

    if (report.context === 'DSA') {
      // STR gains based on rank
      if (report.meta?.rank === 'A') strGain = 3;
      if (report.meta?.rank === 'S') strGain = 5;

      // INT gains for DSA (problem-solving intelligence!)
      if (report.meta?.rank === 'S') intGain = 3;
      else if (report.meta?.rank === 'A') intGain = 2;
      else intGain = 1;

      // SPD based on time taken
      if (report.timeTaken <= (report.meta?.limit || 20) / 2) spdGain = 3;
      else if (report.timeTaken <= (report.meta?.limit || 20)) spdGain = 1;
      else spdGain = -1;
    } else if (report.context === 'BEHAVIORAL') {
      if (report.reflection === 'STAR') intGain = 5;
      else if (report.reflection === 'Detailed') intGain = 3;
      else intGain = 1;
    } else {
      // ML/MATH
      if (report.understanding === 'Deep') intGain = 5;
      else intGain = 2;
      strGain = 2;
    }

    // Tactical retreat bonus (learning from mistakes = INT)
    if (report.retreat) {
      intGain += 3;
      spdGain = 0;
    }

    const newCompleted = {
      ...state.completed,
      [pid]: {
        completed: true,
        retreat: report.retreat,
        notes: report.notes,
        timestamp: Date.now(),
        dateSolved: new Date().toLocaleDateString(),
        timeComplexity: report.timeComplexity,
        spaceComplexity: report.spaceComplexity,
        context: report.context,
        timeTaken: report.timeTaken,
        lastReviewed: Date.now(),
        reviewCount: 0
      }
    };

    const newLegions = { ...state.legions };
    if (report.context !== 'BEHAVIORAL' && report.meta?.type) {
      newLegions[report.meta.type] = (newLegions[report.meta.type] || 0) + 1;
    }

    const newShadows = [...(state.unlockedShadows || [])];
    if (report.meta?.shadow && !report.retreat) {
      if (!newShadows.find(s => s.name === report.meta.shadow)) {
        newShadows.push({
          name: report.meta.shadow,
          rank: report.meta.rank,
          origin: report.title
        });
        alert(`ARISEN: ${report.meta.shadow} (Rank ${report.meta.rank})`);
      }
    }

    const newXp = (state.xp || 0) + (strGain + intGain + Math.max(0, spdGain)) * 10;

    const newState = {
      ...state,
      stats: {
        str: state.stats.str + strGain,
        int: Math.max(0, state.stats.int + intGain),
        spd: Math.max(0, state.stats.spd + spdGain)
      },
      xp: newXp,
      legions: newLegions,
      completed: newCompleted,
      unlockedShadows: newShadows
      // Removed auto-save to saveSlots - DB handles persistence, slots are for manual checkpoints
    };

    setState(newState);
    await saveStateToDB(newState);
    setHuntResult(null);
  };

  const handleSaveSlot = async (slot) => {
    if (!confirm('Create Manual Checkpoint?')) return;

    const newSaveSlots = [...(state.saveSlots || [null, null, null])];
    newSaveSlots[slot] = {
      date: new Date().toLocaleString(),
      data: JSON.stringify(state)
    };

    const newState = { ...state, saveSlots: newSaveSlots };
    setState(newState);
    await saveStateToDB(newState);
  };

  const handleLoadSlot = async (slot) => {
    const save = state.saveSlots?.[slot];
    if (!save) return;
    if (!confirm('REVERT WORLD LINE? All progress since this save will be lost.')) return;

    try {
      const loadedState = JSON.parse(save.data);
      setState({ ...DEFAULT_STATE, ...loadedState });
      await saveStateToDB(loadedState);
    } catch (err) {
      console.error('Failed to load save:', err);
    }
  };

  const handleDownloadBackup = () => {
    const dataStr = JSON.stringify(state);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const filename = `Shadow_Monarch_Backup_${new Date().toISOString().slice(0, 10)}.json`;

    const link = document.createElement('a');
    link.href = dataUri;
    link.download = filename;
    link.click();
    alert('SOUL EXTRACTED SAFELY.');
  };

  const handleUploadBackup = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const content = event.target.result;
        const parsed = JSON.parse(content);
        if (!parsed.stats || !parsed.legions) {
          alert('INVALID SOUL FRAGMENT. File corrupted.');
          return;
        }
        if (confirm('WARNING: REINCARNATION WILL OVERWRITE CURRENT TIMELINE. PROCEED?')) {
          setState({ ...DEFAULT_STATE, ...parsed });
          await saveStateToDB(parsed);
          alert('SOUL RESTORED SUCCESSFULLY.');
        }
      } catch (err) {
        alert('FAILED TO READ SOUL FILE.');
        console.error(err);
      }
    };
    reader.readAsText(file);
  };

  const handleReset = async () => {
    if (!confirm('NUKE ALL PROGRESS? This cannot be undone.')) return;

    try {
      await api.resetAll();
      setState(DEFAULT_STATE);
      alert('ALL DATA RESET.');
    } catch (err) {
      console.error('Failed to reset:', err);
    }
  };

  const handleMarkReviewed = async (problemId) => {
    const problem = state.completed[problemId];
    if (!problem) return;

    const newCompleted = {
      ...state.completed,
      [problemId]: {
        ...problem,
        reviewCount: (problem.reviewCount || 0) + 1,
        lastReviewed: Date.now()
      }
    };

    const newState = { ...state, completed: newCompleted };
    setState(newState);
    await saveStateToDB(newState);
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        fontSize: '24px',
        color: 'var(--neon-blue)'
      }}>
        INITIALIZING SYSTEM...
      </div>
    );
  }

  return (
    <ThemeProvider>
      <div className="container" style={{ paddingBottom: '80px' }}>
        <Header stats={state.stats} xp={state.xp} countdown={countdown} />

        <div className="grid grid-12">
          <QuestContainer
            completed={state.completed}
            onStartHunt={handleStartHunt}
            currentContext={currentContext}
            setCurrentContext={setCurrentContext}
            onOpenStudy={(topic) => setStudyTopic(topic)}
          />

          <Sidebar
            state={state}
            onSaveSlot={handleSaveSlot}
            onLoadSlot={handleLoadSlot}
            onDownloadBackup={handleDownloadBackup}
            onUploadBackup={handleUploadBackup}
            onReset={handleReset}
            onMarkReviewed={handleMarkReviewed}
          />
        </div>

        {/* Hidden file input for backup upload */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".json"
          onChange={handleFileUpload}
        />

        {/* Save Indicator */}
        {showSaveIndicator && (
          <div className="save-indicator">✓ SAVED</div>
        )}

        {/* Hunt Modal */}
        <HuntModal
          isOpen={!!activeHunt}
          hunt={activeHunt}
          onComplete={handleHuntComplete}
          onCancel={handleHuntCancel}
          currentContext={currentContext}
        />

        {/* Report Modal */}
        <ReportModal
          isOpen={!!huntResult}
          hunt={huntResult}
          onSubmit={handleReportSubmit}
          onCancel={() => setHuntResult(null)}
          currentContext={currentContext}
        />

        {/* Study Modal */}
        <StudyModal
          isOpen={!!studyTopic}
          topic={studyTopic}
          content={studyTopic ? getTopicContent(studyTopic) : null}
          onClose={() => setStudyTopic(null)}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;
