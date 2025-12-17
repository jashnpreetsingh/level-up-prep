import { useState } from 'react';

export default function ReportModal({
    isOpen,
    hunt,
    onSubmit,
    onCancel,
    currentContext
}) {
    const [timeComplexity, setTimeComplexity] = useState('O(N)');
    const [spaceComplexity, setSpaceComplexity] = useState('O(1)');
    const [understanding, setUnderstanding] = useState('Deep');
    const [reflection, setReflection] = useState('STAR');

    if (!isOpen || !hunt) return null;

    const handleSubmit = () => {
        const reportData = {
            ...hunt,
            timeComplexity: currentContext === 'DSA' ? timeComplexity : null,
            spaceComplexity: currentContext === 'DSA' ? spaceComplexity : null,
            understanding: currentContext === 'ML' || currentContext === 'MATH' ? understanding : null,
            reflection: currentContext === 'BEHAVIORAL' ? reflection : null
        };
        onSubmit(reportData);
    };

    return (
        <div className="modal-overlay" onClick={onCancel}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
                    HUNT REPORT
                </h3>

                {currentContext === 'DSA' && (
                    <>
                        <div style={{ marginBottom: '16px' }}>
                            <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                                YOUR TIME COMPLEXITY
                            </label>
                            <select
                                value={timeComplexity}
                                onChange={(e) => setTimeComplexity(e.target.value)}
                                style={{ width: '100%' }}
                            >
                                <option value="O(1)">O(1)</option>
                                <option value="O(logN)">O(log N)</option>
                                <option value="O(N)">O(N)</option>
                                <option value="O(NlogN)">O(N log N)</option>
                                <option value="O(N^2)">O(N²)</option>
                            </select>
                        </div>
                        <div style={{ marginBottom: '24px' }}>
                            <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                                YOUR SPACE COMPLEXITY
                            </label>
                            <select
                                value={spaceComplexity}
                                onChange={(e) => setSpaceComplexity(e.target.value)}
                                style={{ width: '100%' }}
                            >
                                <option value="O(1)">O(1)</option>
                                <option value="O(N)">O(N)</option>
                                <option value="O(N^2)">O(N²)</option>
                            </select>
                        </div>
                    </>
                )}

                {(currentContext === 'ML' || currentContext === 'MATH') && (
                    <div style={{ marginBottom: '24px' }}>
                        <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                            UNDERSTANDING DEPTH
                        </label>
                        <select
                            value={understanding}
                            onChange={(e) => setUnderstanding(e.target.value)}
                            style={{ width: '100%' }}
                        >
                            <option value="Surface">Surface Level</option>
                            <option value="Deep">Deep Understanding</option>
                        </select>
                    </div>
                )}

                {currentContext === 'BEHAVIORAL' && (
                    <div style={{ marginBottom: '24px' }}>
                        <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                            REFLECTION QUALITY
                        </label>
                        <select
                            value={reflection}
                            onChange={(e) => setReflection(e.target.value)}
                            style={{ width: '100%' }}
                        >
                            <option value="Basic">Basic Reflection</option>
                            <option value="Detailed">Detailed with Examples</option>
                            <option value="STAR">STAR Format Complete</option>
                        </select>
                    </div>
                )}

                <button
                    className="btn btn-primary"
                    onClick={handleSubmit}
                    style={{ width: '100%', padding: '12px', background: 'var(--neon-purple)', borderColor: 'var(--neon-purple)' }}
                >
                    ABSORB SHADOW
                </button>
            </div>
        </div>
    );
}
