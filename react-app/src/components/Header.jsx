import { useTheme } from '../context/ThemeContext';

export default function Header({ stats, xp, countdown }) {
    const { theme, toggleTheme } = useTheme();
    const level = Math.floor((xp || 0) / 100) + 1;

    return (
        <>
            {/* Theme Toggle */}
            <button className="theme-toggle" onClick={toggleTheme} title="Toggle Theme">
                {theme === 'dark' ? '☀️' : '🌙'}
            </button>

            <header style={{
                borderBottom: '1px solid var(--border-color)',
                paddingBottom: '24px',
                marginBottom: '24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-end',
                flexWrap: 'wrap',
                gap: '24px'
            }}>
                <div>
                    <div style={{ fontSize: '10px', color: 'var(--text-secondary)', letterSpacing: '0.3em', marginBottom: '4px' }}>
                        PLAYER: SHADOW MONARCH
                    </div>
                    <h1 className="glitch" style={{ fontSize: '36px', fontStyle: 'italic', letterSpacing: '-1px' }}>
                        SYSTEM
                    </h1>

                    <div style={{ display: 'flex', gap: '16px', marginTop: '24px' }}>
                        <div className="stat-hex str">
                            <span style={{ fontSize: '9px', fontWeight: 'bold' }}>STR</span>
                            <span style={{ fontSize: '18px', fontWeight: 900 }}>{stats?.str || 10}</span>
                        </div>
                        <div className="stat-hex int">
                            <span style={{ fontSize: '9px', fontWeight: 'bold' }}>INT</span>
                            <span style={{ fontSize: '18px', fontWeight: 900 }}>{stats?.int || 10}</span>
                        </div>
                        <div className="stat-hex spd">
                            <span style={{ fontSize: '9px', fontWeight: 'bold' }}>SPD</span>
                            <span style={{ fontSize: '18px', fontWeight: 900 }}>{stats?.spd || 10}</span>
                        </div>
                    </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                        GOOGLE GATE OPENS IN
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: 'bold', color: 'var(--neon-red)', fontFamily: 'monospace' }}>
                        {countdown}
                    </div>
                    <div style={{ marginTop: '16px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>SYSTEM LEVEL</div>
                        <div style={{ fontSize: '20px', color: 'var(--neon-blue)', fontWeight: 'bold' }}>{level}</div>
                    </div>
                </div>
            </header>
        </>
    );
}
