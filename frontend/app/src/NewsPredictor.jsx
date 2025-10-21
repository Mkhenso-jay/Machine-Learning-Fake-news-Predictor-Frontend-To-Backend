import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    NewspaperIcon,
    ClockIcon,
    TrashIcon,
    CheckCircleIcon,
    XCircleIcon,
    ArrowPathIcon,
    DocumentArrowDownIcon,
    ChartBarIcon,
    ShieldCheckIcon,
    ExclamationTriangleIcon,
    UserGroupIcon,
    CogIcon,
    HomeIcon,
    InformationCircleIcon,
    Bars3Icon,
    XMarkIcon
} from '@heroicons/react/24/outline';

const NewsPredictor = () => {
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [activeTab, setActiveTab] = useState('analyze');
    const [exportFormat, setExportFormat] = useState('pdf');
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const API_URL = 'https://machine-learning-fake-news-detector-api.onrender.com/predict';

    const [backendHealthy, setBackendHealthy] = useState(false);

    const callPredict = async (textToAnalyze) => {
        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textToAnalyze })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => null);
            throw new Error(err?.detail ?? `HTTP ${res.status}`);
        }
        return res.json();
    };

    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch('http://localhost:8000/health');
                if (!res.ok) throw new Error('health check failed');
                const data = await res.json();
                setBackendHealthy(!!data.model_loaded);
            } catch (err) {
                console.warn('Backend health check failed', err);
                setBackendHealthy(false);
            }
        };
        checkHealth();
    }, []);

    // Load history from sessionStorage on component mount
    useEffect(() => {
        const savedHistory = sessionStorage.getItem('newsPredictionHistory');
        if (savedHistory) {
            const parsedHistory = JSON.parse(savedHistory);
            const filteredHistory = parsedHistory.filter(item =>
                Date.now() - item.timestamp < 5 * 60 * 1000
            );
            setHistory(filteredHistory);
            sessionStorage.setItem('newsPredictionHistory', JSON.stringify(filteredHistory));
        }
    }, []);

    // Save history to sessionStorage whenever it changes
    useEffect(() => {
        sessionStorage.setItem('newsPredictionHistory', JSON.stringify(history));
    }, [history]);

    // Clean up old entries every minute
    useEffect(() => {
        const interval = setInterval(() => {
            setHistory(prev => {
                const filtered = prev.filter(item =>
                    Date.now() - item.timestamp < 5 * 60 * 1000
                );
                if (filtered.length !== prev.length) {
                    sessionStorage.setItem('newsPredictionHistory', JSON.stringify(filtered));
                }
                return filtered;
            });
        }, 60000);

        return () => clearInterval(interval);
    }, []);

    const handlePredict = async () => {
        if (!text.trim()) return;

        setLoading(true);
        try {
            const data = await callPredict(text.trim());

            const prediction = {
                id: Date.now(),
                text: text.trim(),
                result: data,
                timestamp: Date.now(),
                exportId: `NEWS-${Date.now()}`
            };

            setResult(data);
            setHistory(prev => [prediction, ...prev.slice(0, 19)]); // Keep last 20 entries
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to get prediction. Make sure your backend is running on port 8000.');
        } finally {
            setLoading(false);
        }
    };

    const clearHistory = () => {
        setHistory([]);
        sessionStorage.removeItem('newsPredictionHistory');
    };

    const formatTime = (timestamp) => {
        const diff = Date.now() - timestamp;
        const minutes = Math.floor(diff / 60000);
        if (minutes < 1) return 'Just now';
        return `${minutes}m ago`;
    };

    const exportResult = (item, format) => {
        const data = {
            analysisId: item.exportId,
            timestamp: new Date(item.timestamp).toISOString(),
            analyzedText: item.text.substring(0, 200) + '...',
            classification: item.result.prediction === 0 ? 'REAL_NEWS' : 'FAKE_NEWS',
            confidence: (item.result.confidence * 100).toFixed(1) + '%',
            probabilities: {
                real: (item.result.probabilities.real * 100).toFixed(1) + '%',
                fake: (item.result.probabilities.fake * 100).toFixed(1) + '%'
            },
            analysisTool: 'TrustNet AI News Verifier',
            disclaimer: 'This analysis is generated by AI and should be used as supporting evidence only.'
        };

        if (format === 'json') {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `news-analysis-${item.exportId}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } else {
            const printWindow = window.open('', '_blank');
            printWindow.document.write(`
        <html>
          <head>
            <title>News Analysis Report - ${item.exportId}</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
              .header { border-bottom: 2px solid #1e40af; padding-bottom: 20px; margin-bottom: 30px; }
              .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
              .real { background: #dcfce7; color: #166534; }
              .fake { background: #fee2e2; color: #991b1b; }
              .section { margin-bottom: 25px; }
            </style>
          </head>
          <body>
            <div class="header">
              <h1>📰 TrustNet AI News Analysis Report</h1>
              <p>Analysis ID: ${data.analysisId}</p>
              <p>Generated: ${new Date(data.timestamp).toLocaleString()}</p>
            </div>
            
            <div class="section">
              <h2>Classification Result</h2>
              <span class="badge ${data.classification === 'REAL_NEWS' ? 'real' : 'fake'}">
                ${data.classification === 'REAL_NEWS' ? '✅ VERIFIED REAL NEWS' : '❌ DETECTED AS FAKE NEWS'}
              </span>
            </div>

            <div class="section">
              <h2>Confidence Metrics</h2>
              <p><strong>Overall Confidence:</strong> ${data.confidence}</p>
              <p><strong>Real Probability:</strong> ${data.probabilities.real}</p>
              <p><strong>Fake Probability:</strong> ${data.probabilities.fake}</p>
            </div>

            <div class="section">
              <h2>Analyzed Content</h2>
              <p style="background: #f3f4f6; padding: 15px; border-radius: 8px;">${data.analyzedText}</p>
            </div>

            <div class="section">
              <p><em>${data.disclaimer}</em></p>
            </div>
          </body>
        </html>
      `);
            printWindow.document.close();
            printWindow.print();
        }
    };

    const getStats = () => {
        const total = history.length;
        const real = history.filter(item => item.result.prediction === 0).length;
        const fake = history.filter(item => item.result.prediction === 1).length;
        return { total, real, fake };
    };

    const stats = getStats();

    // Mobile menu component
    const MobileMenu = () => (
        <AnimatePresence>
            {mobileMenuOpen && (
                <motion.div
                    initial={{ opacity: 0, x: '100%' }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: '100%' }}
                    className="fixed inset-0 z-50 lg:hidden"
                >
                    <div className="fixed inset-0 bg-black/30" onClick={() => setMobileMenuOpen(false)} />
                    <div className="fixed top-0 right-0 w-64 h-full bg-white shadow-xl p-6">
                        <div className="flex justify-between items-center mb-8">
                            <h2 className="text-lg font-bold">Menu</h2>
                            <button onClick={() => setMobileMenuOpen(false)}>
                                <XMarkIcon className="h-6 w-6" />
                            </button>
                        </div>
                        <nav className="space-y-4">
                            {[
                                { id: 'analyze', name: 'Analyze News', icon: HomeIcon },
                                { id: 'history', name: 'Case History', icon: ClockIcon },
                                { id: 'stats', name: 'Statistics', icon: ChartBarIcon }
                            ].map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => {
                                        setActiveTab(item.id);
                                        setMobileMenuOpen(false);
                                    }}
                                    className={`flex items-center space-x-3 w-full px-3 py-2 rounded-lg transition-all duration-200 ${activeTab === item.id
                                        ? 'bg-red-50 text-red-600 border border-red-200'
                                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                                        }`}
                                >
                                    <item.icon className="h-5 w-5" />
                                    <span>{item.name}</span>
                                </button>
                            ))}
                        </nav>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
            {/* Header */}
            <motion.header
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-40"
            >
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-4">
                        <div className="flex items-center space-x-3">
                            <div className="bg-red-600 p-2 rounded-xl">
                                <ShieldCheckIcon className="h-6 w-6 sm:h-8 sm:w-8 text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl sm:text-2xl font-bold text-slate-900">Fake News Detector</h1>
                                <p className="text-xs sm:text-sm text-slate-600">News Verification AI</p>
                            </div>
                        </div>

                        {/* Desktop Navigation */}
                        <nav className="hidden lg:flex space-x-8">
                            {[
                                { id: 'analyze', name: 'Analyze News', icon: HomeIcon },
                                { id: 'history', name: 'Case History', icon: ClockIcon },
                                { id: 'stats', name: 'Statistics', icon: ChartBarIcon }
                            ].map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => setActiveTab(item.id)}
                                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 ${activeTab === item.id
                                        ? 'bg-red-50 text-red-600 border border-red-200'
                                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                                        }`}
                                >
                                    <item.icon className="h-4 w-4" />
                                    <span className="hidden sm:inline">{item.name}</span>
                                </button>
                            ))}
                        </nav>

                        {/* Mobile Menu Button */}
                        <button 
                            onClick={() => setMobileMenuOpen(true)}
                            className="lg:hidden p-2 rounded-lg hover:bg-slate-100 transition-colors"
                        >
                            <Bars3Icon className="h-6 w-6 text-slate-600" />
                        </button>
                    </div>
                </div>
            </motion.header>

            <MobileMenu />

            <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8 py-4 sm:py-8">
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 sm:gap-6 lg:gap-8">
                    {/* Sidebar - Hidden on mobile, shown on lg+ */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="hidden lg:block lg:col-span-1 space-y-6"
                    >
                        {/* Quick Stats */}
                        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-6">
                            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                                <ChartBarIcon className="h-5 w-5" />
                                Session Overview
                            </h3>
                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <span className="text-slate-600">Total Analysis</span>
                                    <span className="font-bold text-slate-900">{stats.total}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-green-600">Verified Real</span>
                                    <span className="font-bold text-green-600">{stats.real}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-red-600">Detected Fake</span>
                                    <span className="font-bold text-red-600">{stats.fake}</span>
                                </div>
                            </div>
                        </div>

                        {/* Use Cases */}
                        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 sm:p-6">
                            <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                                <UserGroupIcon className="h-5 w-5" />
                                For Professionals
                            </h3>
                            <div className="space-y-3 text-sm">
                                <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                    <strong>Journalists:</strong> Verify sources before publishing
                                </div>
                                <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                                    <strong>Researchers:</strong> Analyze misinformation patterns
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* Main Content */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="col-span-1 lg:col-span-3 space-y-6 sm:space-y-8"
                    >
                        {/* Mobile Stats Bar - Only shown on mobile */}
                        <div className="lg:hidden bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
                            <div className="grid grid-cols-3 gap-4 text-center">
                                <div>
                                    <div className="text-lg sm:text-xl font-bold text-slate-900">{stats.total}</div>
                                    <div className="text-xs text-slate-600">Total</div>
                                </div>
                                <div>
                                    <div className="text-lg sm:text-xl font-bold text-green-600">{stats.real}</div>
                                    <div className="text-xs text-green-600">Real</div>
                                </div>
                                <div>
                                    <div className="text-lg sm:text-xl font-bold text-red-600">{stats.fake}</div>
                                    <div className="text-xs text-red-600">Fake</div>
                                </div>
                            </div>
                        </div>

                        {activeTab === 'analyze' && (
                            <>
                                {/* Input Section */}
                                <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-4 sm:p-6 w-full">
                                    <div className="flex items-center justify-between mb-4 sm:mb-6">
                                        <h3 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
                                            <NewspaperIcon className="h-5 w-5 sm:h-6 sm:w-6" />
                                            Analyze News Content
                                        </h3>
                                        <div className="text-xs sm:text-sm text-slate-500">
                                            {text.length}/5000
                                        </div>
                                    </div>

                                    <textarea
                                        value={text}
                                        onChange={(e) => setText(e.target.value)}
                                        placeholder="Paste news article, social media post, or any text content you want to verify..."
                                        className="w-full h-48 sm:h-64 px-4 py-4 border border-red-300 rounded-xl resize-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all duration-200 text-slate-700 placeholder-slate-400"
                                        disabled={loading}
                                        maxLength={5000}
                                    />

                                    <motion.button
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        onClick={handlePredict}
                                        disabled={loading || !text.trim()}
                                        className={`w-full mt-4 sm:mt-6 py-3 sm:py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 flex items-center justify-center gap-3 ${loading || !text.trim()
                                            ? 'bg-slate-400 cursor-not-allowed'
                                            : 'bg-red-600 hover:bg-red-700 shadow-lg hover:shadow-xl'
                                            }`}
                                    >
                                        {loading ? (
                                            <>
                                                <ArrowPathIcon className="h-4 w-4 sm:h-5 sm:w-5 animate-spin" />
                                                <span className="text-sm sm:text-base">Analyzing Content...</span>
                                            </>
                                        ) : (
                                            <>
                                                <ShieldCheckIcon className="h-4 w-4 sm:h-5 sm:w-5" />
                                                <span className="text-sm sm:text-base">Verify Authenticity</span>
                                            </>
                                        )}
                                    </motion.button>
                                </div>

                                {/* Results Section */}
                                <AnimatePresence>
                                    {result && (
                                        <motion.div
                                            initial={{ opacity: 0, x: 60 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: 60 }}
                                            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                                            className={`fixed lg:top-20 lg:right-0 lg:h-screen lg:w-96 lg:max-w-full
                                                bottom-0 left-0 right-0 h-3/4 sm:h-2/3
                                                z-50 p-4 sm:p-6 rounded-t-2xl lg:rounded-l-2xl lg:rounded-t-none
                                                shadow-2xl overflow-auto
                                                ${result.prediction === 0 ? 'bg-white/95 border-l-8 border-green-500' : 'bg-white/95 border-l-8 border-red-500'}`}
                                            role="region"
                                            aria-label="Prediction summary"
                                        >
                                            {/* Close button for mobile */}
                                            <button 
                                                onClick={() => setResult(null)}
                                                className="lg:hidden absolute top-4 right-4 p-2 hover:bg-slate-100 rounded-lg transition-colors"
                                            >
                                                <XMarkIcon className="h-5 w-5" />
                                            </button>

                                            <div className="flex items-center justify-between mb-4 sm:mb-6">
                                                <div className="flex items-center gap-3 sm:gap-4">
                                                    {result.prediction === 0 ? (
                                                        <CheckCircleIcon className="h-8 w-8 sm:h-10 sm:w-10 text-green-500" />
                                                    ) : (
                                                        <XCircleIcon className="h-8 w-8 sm:h-10 sm:w-10 text-red-500" />
                                                    )}
                                                    <div>
                                                        <h3 className="text-lg sm:text-2xl font-bold text-slate-900">
                                                            {result.prediction === 0 ? '✅ Verified Real News' : '❌ Detected as Fake News'}
                                                        </h3>
                                                        <p className="text-xs sm:text-sm text-slate-600">Analysis completed with high confidence</p>
                                                    </div>
                                                </div>
                                                <button 
                                                    onClick={() => exportResult(history[0], exportFormat)}
                                                    className="hidden lg:flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-slate-700 transition-colors"
                                                >
                                                    <DocumentArrowDownIcon className="h-4 w-4" />
                                                    Export
                                                </button>
                                            </div>

                                            <div className="grid grid-cols-3 gap-3 sm:gap-6 mb-4 sm:mb-6">
                                                <div className="text-center p-3 sm:p-4 bg-white rounded-xl border border-slate-200">
                                                    <div className="text-lg sm:text-2xl font-bold text-green-600">
                                                        {(result.probabilities.real * 100).toFixed(1)}%
                                                    </div>
                                                    <div className="text-xs sm:text-sm text-slate-600">Real</div>
                                                </div>
                                                <div className="text-center p-3 sm:p-4 bg-white rounded-xl border border-slate-200">
                                                    <div className="text-lg sm:text-2xl font-bold text-red-600">
                                                        {(result.probabilities.fake * 100).toFixed(1)}%
                                                    </div>
                                                    <div className="text-xs sm:text-sm text-slate-600">Fake</div>
                                                </div>
                                                <div className="text-center p-3 sm:p-4 bg-white rounded-xl border border-slate-200">
                                                    <div className="text-lg sm:text-2xl font-bold text-blue-600">
                                                        {(result.confidence * 100).toFixed(1)}%
                                                    </div>
                                                    <div className="text-xs sm:text-sm text-slate-600">Confidence</div>
                                                </div>
                                            </div>

                                            {/* Mobile Export Button */}
                                            <button 
                                                onClick={() => exportResult(history[0], exportFormat)}
                                                className="lg:hidden w-full py-3 bg-red-600 text-white rounded-lg font-semibold flex items-center justify-center gap-2 mb-4"
                                            >
                                                <DocumentArrowDownIcon className="h-4 w-4" />
                                                Export Result
                                            </button>

                                            {result.confidence < 0.7 && (
                                                <motion.div 
                                                    initial={{ opacity: 0 }} 
                                                    animate={{ opacity: 1 }} 
                                                    className="flex items-center gap-3 p-3 sm:p-4 bg-yellow-50 rounded-lg border border-yellow-200"
                                                >
                                                    <ExclamationTriangleIcon className="h-4 w-4 sm:h-5 sm:w-5 text-yellow-600" />
                                                    <div>
                                                        <p className="text-sm sm:text-base text-yellow-800 font-medium">Low Confidence Analysis</p>
                                                        <p className="text-xs sm:text-sm text-yellow-700">
                                                            Add more context or complete article text for better accuracy.
                                                        </p>
                                                    </div>
                                                </motion.div>
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </>
                        )}

                        {activeTab === 'history' && (
                            <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-4 sm:p-6">
                                <div className="flex items-center justify-between mb-4 sm:mb-6">
                                    <h3 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
                                        <ClockIcon className="h-5 w-5 sm:h-6 sm:w-6" />
                                        Case History
                                    </h3>
                                    {history.length > 0 && (
                                        <button
                                            onClick={clearHistory}
                                            className="flex items-center gap-2 px-3 sm:px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors text-sm sm:text-base"
                                        >
                                            <TrashIcon className="h-4 w-4" />
                                            Clear All
                                        </button>
                                    )}
                                </div>

                                <div className="space-y-3 sm:space-y-4">
                                    <AnimatePresence>
                                        {history.length === 0 ? (
                                            <motion.div
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 1 }}
                                                className="text-center py-8 sm:py-12 text-slate-500"
                                            >
                                                <NewspaperIcon className="h-12 w-12 sm:h-16 sm:w-16 mx-auto mb-3 sm:mb-4 text-slate-300" />
                                                <p className="text-base sm:text-lg">No analysis history yet</p>
                                                <p className="text-sm">Start by analyzing some news content</p>
                                            </motion.div>
                                        ) : (
                                            history.map((item) => (
                                                <motion.div
                                                    key={item.id}
                                                    initial={{ opacity: 0, scale: 0.95 }}
                                                    animate={{ opacity: 1, scale: 1 }}
                                                    exit={{ opacity: 0, scale: 0.95 }}
                                                    className={`p-3 sm:p-4 rounded-xl border-l-4 ${item.result.prediction === 0
                                                        ? 'border-green-400 bg-green-50/50'
                                                        : 'border-red-400 bg-red-50/50'
                                                        }`}
                                                >
                                                    <div className="flex justify-between items-start mb-2 sm:mb-3">
                                                        <div className="flex items-center gap-2 sm:gap-3">
                                                            <span className={`font-semibold text-sm sm:text-base ${item.result.prediction === 0 ? 'text-green-700' : 'text-red-700'
                                                                }`}>
                                                                {item.result.prediction === 0 ? '✅ Real' : '❌ Fake'}
                                                            </span>
                                                            <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded">
                                                                {item.exportId}
                                                            </span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-xs text-slate-500">
                                                                {formatTime(item.timestamp)}
                                                            </span>
                                                            <button
                                                                onClick={() => exportResult(item, exportFormat)}
                                                                className="p-1 hover:bg-slate-100 rounded transition-colors"
                                                                title="Export as evidence"
                                                            >
                                                                <DocumentArrowDownIcon className="h-3 w-3 sm:h-4 sm:w-4 text-slate-500" />
                                                            </button>
                                                        </div>
                                                    </div>
                                                    <p className="text-xs sm:text-sm text-slate-700 line-clamp-2 mb-2">
                                                        {item.text}
                                                    </p>
                                                    <div className="flex justify-between items-center text-xs text-slate-500">
                                                        <span>Confidence: {(item.result.confidence * 100).toFixed(1)}%</span>
                                                        <span>Real: {(item.result.probabilities.real * 100).toFixed(1)}%</span>
                                                    </div>
                                                </motion.div>
                                            ))
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>
                        )}

                        {activeTab === 'stats' && (
                            <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-4 sm:p-6">
                                <h3 className="text-lg sm:text-xl font-bold text-slate-900 mb-4 sm:mb-6">Analysis Statistics</h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                                    <div className="text-center p-4 sm:p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border border-blue-200">
                                        <div className="text-2xl sm:text-3xl font-bold text-blue-600">{stats.total}</div>
                                        <div className="text-blue-700 text-sm sm:text-base">Total Analysis</div>
                                    </div>
                                    <div className="text-center p-4 sm:p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-xl border border-green-200">
                                        <div className="text-2xl sm:text-3xl font-bold text-green-600">{stats.real}</div>
                                        <div className="text-green-700 text-sm sm:text-base">Verified Real</div>
                                    </div>
                                    <div className="text-center p-4 sm:p-6 bg-gradient-to-br from-red-50 to-red-100 rounded-xl border border-red-200">
                                        <div className="text-2xl sm:text-3xl font-bold text-red-600">{stats.fake}</div>
                                        <div className="text-red-700 text-sm sm:text-base">Detected Fake</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </motion.div>
                </div>
            </div>
        </div>
    );
};

export default NewsPredictor;