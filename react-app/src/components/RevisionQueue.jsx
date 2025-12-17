import { useMemo } from 'react';

// Spaced repetition intervals in days
const INTERVALS = [1, 3, 7, 14];

export default function RevisionQueue({ completed, onMarkReviewed }) {
    const revisionQueue = useMemo(() => {
        const now = Date.now();
        const queue = [];

        Object.entries(completed || {}).forEach(([id, data]) => {
            // Only DSA problems need revision
            if (data.context !== 'DSA') return;
            if (!data.timestamp) return;

            const daysSinceSolved = Math.floor((now - data.timestamp) / (1000 * 60 * 60 * 24));
            const reviewCount = data.reviewCount || 0;

            // Check if due for review based on spaced repetition
            if (reviewCount < INTERVALS.length) {
                const nextReviewDay = INTERVALS[reviewCount];
                if (daysSinceSolved >= nextReviewDay) {
                    // Extract problem name from ID (e.g., "DSA_Two_Sum" -> "Two Sum")
                    const problemName = id.replace(/^DSA_/, '').replace(/_/g, ' ');

                    queue.push({
                        id,
                        name: problemName,
                        daysSince: daysSinceSolved,
                        reviewCount,
                        nextInterval: INTERVALS[reviewCount + 1] || 'MASTERED',
                        urgency: daysSinceSolved - nextReviewDay
                    });
                }
            }
        });

        // Sort by urgency (most overdue first)
        return queue.sort((a, b) => b.urgency - a.urgency);
    }, [completed]);

    if (revisionQueue.length === 0) {
        return (
            <div className="panel" style={{ padding: '16px' }}>
                <h3 style={{
                    fontSize: '12px',
                    color: 'var(--neon-gold)',
                    fontWeight: 'bold',
                    marginBottom: '8px',
                    letterSpacing: '0.2em',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                }}>
                    🔄 REVISION QUEUE
                </h3>
                <div style={{
                    padding: '16px',
                    border: '1px dashed var(--border-color)',
                    textAlign: 'center',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                    borderRadius: '4px'
                }}>
                    ALL CLEAR — NO REVIEWS DUE
                </div>
            </div>
        );
    }

    return (
        <div className="panel" style={{ padding: '16px' }}>
            <h3 style={{
                fontSize: '12px',
                color: 'var(--neon-gold)',
                fontWeight: 'bold',
                marginBottom: '12px',
                letterSpacing: '0.2em',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
            }}>
                <span>🔄 REVISION QUEUE</span>
                <span style={{
                    background: 'var(--neon-red)',
                    color: '#000',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: 'bold'
                }}>
                    {revisionQueue.length} DUE
                </span>
            </h3>

            <div style={{ fontSize: '10px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                Spaced Repetition: 1d → 3d → 7d → 14d → Mastered
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                {revisionQueue.slice(0, 5).map(item => (
                    <div
                        key={item.id}
                        style={{
                            padding: '8px 12px',
                            background: item.urgency > 3 ? 'rgba(255, 42, 42, 0.1)' : 'var(--card-bg)',
                            border: `1px solid ${item.urgency > 3 ? 'var(--neon-red)' : 'var(--border-color)'}`,
                            borderRadius: '4px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}
                    >
                        <div>
                            <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '2px' }}>
                                {item.name}
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                                {item.daysSince}d ago • Review #{item.reviewCount + 1}
                            </div>
                        </div>
                        <button
                            onClick={() => onMarkReviewed(item.id)}
                            className="btn"
                            style={{
                                fontSize: '9px',
                                padding: '4px 8px',
                                borderColor: 'var(--neon-gold)',
                                color: 'var(--neon-gold)'
                            }}
                        >
                            ✓ DONE
                        </button>
                    </div>
                ))}
            </div>

            {revisionQueue.length > 5 && (
                <div style={{
                    textAlign: 'center',
                    fontSize: '10px',
                    color: 'var(--text-secondary)',
                    marginTop: '8px'
                }}>
                    +{revisionQueue.length - 5} more due...
                </div>
            )}
        </div>
    );
}
