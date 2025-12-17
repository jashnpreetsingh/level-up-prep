import { useState, useEffect, useRef } from 'react';

export default function HuntModal({
    isOpen,
    hunt,
    onComplete,
    onCancel,
    currentContext
}) {
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const [isRetreating, setIsRetreating] = useState(false);
    const [notes, setNotes] = useState('');
    const intervalRef = useRef(null);
    const startTimeRef = useRef(null);
    const pausedTimeRef = useRef(0);

    useEffect(() => {
        if (isOpen && hunt) {
            setElapsedSeconds(0);
            setIsRetreating(false);
            setNotes('');
            startTimeRef.current = Date.now();
            pausedTimeRef.current = 0;

            // Use start time instead of increment to avoid background throttling
            intervalRef.current = setInterval(() => {
                if (!isRetreating && startTimeRef.current) {
                    const now = Date.now();
                    const elapsed = Math.floor((now - startTimeRef.current) / 1000) + pausedTimeRef.current;
                    setElapsedSeconds(elapsed);
                }
            }, 1000);
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [isOpen, hunt]);

    useEffect(() => {
        if (isRetreating) {
            // Pause: save accumulated time
            if (startTimeRef.current) {
                pausedTimeRef.current = elapsedSeconds;
                startTimeRef.current = null;
            }
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        } else if (isOpen && !intervalRef.current) {
            // Resume: start fresh timer
            startTimeRef.current = Date.now();
            intervalRef.current = setInterval(() => {
                if (startTimeRef.current) {
                    const now = Date.now();
                    const elapsed = Math.floor((now - startTimeRef.current) / 1000) + pausedTimeRef.current;
                    setElapsedSeconds(elapsed);
                }
            }, 1000);
        }
    }, [isRetreating, isOpen]);

    if (!isOpen || !hunt) return null;

    const minutes = Math.floor(elapsedSeconds / 60).toString().padStart(2, '0');
    const seconds = (elapsedSeconds % 60).toString().padStart(2, '0');
    const isOvertime = Math.floor(elapsedSeconds / 60) > (hunt.meta?.limit || 20);

    const handleTacticalRetreat = () => {
        setIsRetreating(true);
    };

    const handleReEngage = () => {
        if (notes.length < 5) {
            alert('Enter your notes (at least 5 characters)');
            return;
        }
        // Reset timer for re-engage
        pausedTimeRef.current = 0;
        setElapsedSeconds(0);
        setIsRetreating(false);
    };

    const handleComplete = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        onComplete({
            title: hunt.title,
            context: hunt.context,
            meta: hunt.meta,
            retreat: isRetreating || notes.length > 0,
            notes,
            timeTaken: Math.ceil(elapsedSeconds / 60)
        });
    };

    const handleCancel = () => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        onCancel();
    };

    const getLeetCodeLink = () => {
        if (currentContext !== 'DSA') return null;
        if (hunt.title.startsWith('(')) return null;
        const slug = hunt.title.toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .replace(/\s+/g, '-')
            .replace(/^-+|-+$/g, '');
        return slug ? `https://leetcode.com/problems/${slug}/` : null;
    };

    const leetcodeLink = getLeetCodeLink();

    return (
        <div className="modal-overlay" onClick={handleCancel}>
            <div className="modal-content" onClick={e => e.stopPropagation()} style={{ textAlign: 'center' }}>
                {!isRetreating ? (
                    <>
                        <div style={{ color: 'var(--neon-blue)', fontSize: '12px', letterSpacing: '0.3em', marginBottom: '8px' }} className="pulse">
                            DUNGEON ACTIVE
                        </div>
                        <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
                            {hunt.title}
                        </h2>

                        {leetcodeLink && (
                            <a
                                href={leetcodeLink}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{ color: 'var(--neon-blue)', fontSize: '12px', textDecoration: 'underline', display: 'block', marginBottom: '16px' }}
                            >
                                🔗 Open on LeetCode
                            </a>
                        )}

                        <div className={`timer ${isOvertime ? 'overtime' : ''}`} style={{ marginBottom: '8px' }}>
                            {minutes}:{seconds}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '32px' }}>
                            TARGET: {hunt.meta?.limit || 20}:00
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <button className="btn btn-primary" onClick={handleComplete} style={{ width: '100%', padding: '12px', fontSize: '14px' }}>
                                COMPLETE HUNT
                            </button>
                            <button
                                className="btn"
                                onClick={handleTacticalRetreat}
                                style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                            >
                                🏳️ TACTICAL RETREAT (LEARN)
                            </button>
                            <button
                                onClick={handleCancel}
                                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: '12px', cursor: 'pointer', marginTop: '8px' }}
                            >
                                ABORT
                            </button>
                        </div>
                    </>
                ) : (
                    <>
                        <div style={{ color: 'var(--neon-gold)', fontSize: '12px', letterSpacing: '0.3em', marginBottom: '8px' }}>
                            TACTICAL RETREAT
                        </div>
                        <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '16px' }}>
                            GRIMOIRE (STUDY MODE)
                        </h2>
                        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px', textAlign: 'left' }}>
                            Timer is paused. Study the concept. Enter the "Magic Spell" (Core Insight) below to re-engage.
                        </p>

                        <textarea
                            value={notes}
                            onChange={(e) => setNotes(e.target.value)}
                            placeholder="e.g. Use a monotonic stack to find the next greater element..."
                            style={{
                                width: '100%',
                                height: '120px',
                                resize: 'none',
                                marginBottom: '16px'
                            }}
                        />

                        <button
                            className="btn"
                            onClick={handleReEngage}
                            style={{
                                width: '100%',
                                padding: '12px',
                                background: 'rgba(255, 215, 0, 0.1)',
                                borderColor: 'var(--neon-gold)',
                                color: 'var(--neon-gold)'
                            }}
                        >
                            ⚔️ RE-ENGAGE BOSS
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}
