import RevisionQueue from './RevisionQueue';

export default function Sidebar({
    state,
    onSaveSlot,
    onLoadSlot,
    onDownloadBackup,
    onUploadBackup,
    onReset,
    onMarkReviewed
}) {
    const { stats, legions, unlockedShadows, saveSlots, completed, xp } = state;

    // Calculate analytics
    const completedCount = Object.keys(completed || {}).length;
    const todayStr = new Date().toDateString();
    const todayCount = Object.values(completed || {}).filter(c => {
        const ts = c.timestamp;
        return ts && new Date(ts).toDateString() === todayStr;
    }).length;

    // Calculate streak
    const calculateStreak = () => {
        const dates = Object.values(completed || {})
            .map(c => c.timestamp ? new Date(c.timestamp).toDateString() : null)
            .filter(d => d)
            .filter((v, i, a) => a.indexOf(v) === i)
            .sort((a, b) => new Date(b) - new Date(a));

        if (dates.length === 0) return 0;

        let streak = 0;
        let checkDate = new Date();

        if (!dates.includes(checkDate.toDateString())) {
            checkDate.setDate(checkDate.getDate() - 1);
        }

        while (dates.includes(checkDate.toDateString())) {
            streak++;
            checkDate.setDate(checkDate.getDate() - 1);
        }

        return streak;
    };

    const streak = calculateStreak();
    const labels = ["AUTO-SAVE", "SLOT 1", "SLOT 2"];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* Save Slots */}
            <div className="panel" style={{ padding: '16px' }}>
                <h3 style={{ fontSize: '12px', color: 'var(--neon-blue)', fontWeight: 'bold', marginBottom: '12px', letterSpacing: '0.2em' }}>
                    CHRONOS STONE (SAVES)
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {(saveSlots || [null, null, null]).map((slot, idx) => (
                        <div
                            key={idx}
                            style={{
                                padding: '8px',
                                background: 'var(--card-bg)',
                                border: `1px ${slot ? 'solid' : 'dashed'} var(--border-color)`,
                                borderRadius: '4px',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                fontSize: '10px',
                                opacity: slot ? 1 : 0.5,
                                cursor: slot ? 'pointer' : 'default'
                            }}
                            onClick={() => slot && onLoadSlot(idx)}
                        >
                            <div>
                                <span style={{ color: 'var(--neon-blue)', fontWeight: 'bold' }}>{labels[idx]}</span>
                                {slot && <div style={{ color: 'var(--text-secondary)', marginTop: '2px' }}>{slot.date}</div>}
                                {!slot && <span style={{ marginLeft: '8px' }}>(EMPTY)</span>}
                            </div>
                            {slot && <span style={{ color: 'var(--neon-blue)' }}>LOAD ↺</span>}
                        </div>
                    ))}
                </div>
                <button
                    className="btn"
                    onClick={() => onSaveSlot(1)}
                    style={{ width: '100%', marginTop: '8px', fontSize: '10px' }}
                >
                    + CREATE MANUAL CHECKPOINT
                </button>
            </div>

            {/* Soul Crystal (Backup) */}
            <div className="panel" style={{ padding: '16px' }}>
                <h3 style={{ fontSize: '12px', color: 'var(--neon-gold)', fontWeight: 'bold', marginBottom: '8px', letterSpacing: '0.2em' }}>
                    🔮 SOUL CRYSTAL (BACKUP)
                </h3>
                <p style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                    <span style={{ color: 'var(--neon-blue)' }}>Using SQLite Database</span><br />
                    Extract your soul to a file to prevent permanent death.
                </p>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                        className="btn"
                        onClick={onDownloadBackup}
                        style={{ flex: 1, fontSize: '10px', borderColor: 'rgba(255, 215, 0, 0.5)', color: 'var(--neon-gold)' }}
                    >
                        ⬇ EXTRACT
                    </button>
                    <button
                        className="btn"
                        onClick={onUploadBackup}
                        style={{ flex: 1, fontSize: '10px', borderColor: 'rgba(0, 102, 204, 0.5)', color: 'var(--neon-blue)' }}
                    >
                        ⬆ UPLOAD
                    </button>
                </div>
            </div>

            {/* Revision Queue */}
            <RevisionQueue completed={completed} onMarkReviewed={onMarkReviewed} />

            {/* Shadow Army */}
            <div>
                <h2 style={{
                    color: 'var(--neon-purple)',
                    fontWeight: 'bold',
                    fontSize: '18px',
                    letterSpacing: '0.2em',
                    borderLeft: '4px solid var(--neon-purple)',
                    paddingLeft: '12px',
                    marginBottom: '16px'
                }}>
                    SHADOW ARMY
                </h2>

                {(!unlockedShadows || unlockedShadows.length === 0) ? (
                    <div style={{
                        padding: '16px',
                        border: '1px dashed var(--border-color)',
                        textAlign: 'center',
                        fontSize: '12px',
                        color: 'var(--text-secondary)',
                        borderRadius: '4px'
                    }}>
                        NO COMMANDERS ARISEN
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {unlockedShadows.map((shadow, idx) => (
                            <div key={idx} className="shadow-card" style={{ padding: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontSize: '10px', color: 'var(--text-secondary)', fontWeight: 'bold', letterSpacing: '0.1em' }}>COMMANDER</div>
                                    <div style={{ fontSize: '14px', fontWeight: 900 }}>{shadow.name}</div>
                                </div>
                                <div style={{ fontSize: '20px', fontWeight: 900, color: shadow.rank === 'S' ? 'var(--neon-gold)' : 'var(--neon-purple)' }}>
                                    {shadow.rank}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Legion Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '16px' }}>
                    {Object.entries(legions || {}).filter(([_, count]) => count > 0).map(([type, count]) => {
                        const icons = { Mage: '⚡', Tank: '🛡', Assassin: '🗡', Infantry: '♟', Construct: '🤖', Rune: '📜' };
                        return (
                            <div key={type} style={{
                                background: 'var(--card-bg)',
                                padding: '8px',
                                border: '1px solid var(--border-color)',
                                borderRadius: '4px',
                                textAlign: 'center'
                            }}>
                                <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{type.toUpperCase()}</div>
                                <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{icons[type] || '♟'} x{count}</div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Analytics */}
            <div className="panel" style={{ padding: '16px' }}>
                <h3 style={{ fontSize: '12px', color: 'var(--neon-blue)', fontWeight: 'bold', marginBottom: '12px', letterSpacing: '0.2em' }}>
                    📊 PROGRESS ANALYTICS
                </h3>
                <div style={{ fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Problems Completed</span>
                        <span style={{ color: 'var(--neon-blue)', fontWeight: 'bold' }}>{completedCount}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Daily Streak</span>
                        <span style={{ color: 'var(--neon-gold)', fontWeight: 'bold' }}>{streak} days</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Problems Today</span>
                        <span style={{ color: '#00cc66', fontWeight: 'bold' }}>{todayCount}</span>
                    </div>
                </div>
            </div>

            {/* Reset */}
            <div className="panel" style={{ padding: '16px' }}>
                <h3 style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 'bold', marginBottom: '8px' }}>
                    SYSTEM CONTROLS
                </h3>
                <button className="btn btn-danger" onClick={onReset} style={{ width: '100%' }}>
                    RESET ALL DATA
                </button>
            </div>
        </div>
    );
}
