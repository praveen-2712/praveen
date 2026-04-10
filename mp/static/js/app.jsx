const { useState, useRef } = React;

function App() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const fileInput = useRef(null);

    const handleFileChange = (e) => {
        const f = e.target.files[0];
        if (f) {
            setFile(f);
            setPreview(URL.createObjectURL(f));
            setResults(null);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            setResults(data);
        } catch (err) {
            console.error(err);
            alert("Error analyzing image.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen p-8 bg-slate-900 text-slate-100 flex flex-col items-center">
            <h1 className="text-4xl font-extrabold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-500">
                Neurologix AI
            </h1>
            <p className="text-slate-400 mb-8 text-lg">Research-Grade Brain Tumor Detection Pipeline</p>

            <div className="flex flex-col md:flex-row gap-8 w-full max-w-7xl">
                {/* Left Panel */}
                <div className="flex-1 space-y-6">
                    <div 
                        className="glass-panel p-8 text-center upload-zone cursor-pointer"
                        onClick={() => fileInput.current.click()}
                    >
                        <input type="file" ref={fileInput} className="hidden" accept="image/*" onChange={handleFileChange} />
                        {preview ? (
                            <img src={preview} alt="MRI" className="max-h-96 mx-auto rounded-lg shadow-lg border border-slate-700" />
                        ) : (
                            <div className="py-12 text-slate-400">
                                <svg className="w-16 h-16 mx-auto mb-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                                <p className="text-xl font-medium">Click here to upload an MRI scan (.jpg, .png)</p>
                            </div>
                        )}
                    </div>

                    <button 
                        onClick={handleAnalyze} 
                        disabled={!file || loading}
                        className={`w-full py-4 rounded-xl font-bold text-lg shadow-xl shadow-indigo-500/20 transition-all ${!file ? 'bg-slate-800 text-slate-500' : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500'}`}
                    >
                        {loading ? (
                            <div className="flex justify-center items-center gap-3">
                                <div className="loader border-t-transparent w-6 h-6 border-2"></div>
                                Running Pipeline...
                            </div>
                        ) : "Execute Hybrid AI Pipeline"}
                    </button>
                    
                    {results && (
                        <div className="glass-panel p-6 space-y-4">
                            <h2 className="text-2xl font-bold text-indigo-400 flex items-center gap-2">
                                <span className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse"></span>
                                Diagnostic Report
                            </h2>
                            <p className="font-semibold text-lg text-slate-200">{results.report.summary}</p>
                            <p className="text-slate-400 leading-relaxed text-sm">{results.report.details}</p>
                            <div className="mt-4 p-4 bg-slate-800/50 rounded-lg border-l-4 border-yellow-500">
                                <p className="text-xs font-bold text-yellow-500 uppercase tracking-wider mb-1">Recommendation</p>
                                <p className="text-slate-300 text-sm">{results.report.recommendation}</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel */}
                <div className="flex-[1.5] space-y-6">
                    {results ? (
                        <>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="glass-panel p-4">
                                    <h3 className="text-center font-bold mb-3 text-indigo-300 flex items-center justify-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"></path></svg>
                                        Detection (YOLOv8)
                                    </h3>
                                    <img src={results.boxes} className="w-full h-80 object-contain rounded-lg bg-black" alt="Boxes" />
                                </div>
                                <div className="glass-panel p-4">
                                    <h3 className="text-center font-bold mb-3 text-teal-300 flex items-center justify-center gap-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                        Segmentation (U-Net)
                                    </h3>
                                    <img src={results.segmented} className="w-full h-80 object-contain rounded-lg bg-black" alt="Segmented" />
                                </div>
                            </div>
                            
                            <h3 className="text-2xl font-bold mt-8 border-b border-slate-700/50 pb-2 flex items-center gap-2">
                                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                                Isolated Pathologies
                            </h3>
                            <div className="grid grid-cols-1 gap-4">
                                {results.tumors.map(t => (
                                    <div key={t.id} className="glass-panel p-5 flex flex-col md:flex-row gap-4">
                                        <div className="w-full md:w-1/3 border-r border-slate-700/50 pr-4 flex flex-col justify-center">
                                            <span className="font-bold text-xl text-purple-300 mb-1">Tumor #{t.id}</span>
                                            <div className="bg-slate-800 p-3 rounded-lg mt-2 space-y-2">
                                                <div className="flex justify-between items-center">
                                                    <span className="text-slate-400 text-sm">Classification:</span>
                                                    <span className="bg-red-500/20 text-red-300 text-xs px-2 py-1 rounded font-bold border border-red-500/30">
                                                        {t.type}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between items-center">
                                                    <span className="text-slate-400 text-sm">Confidence:</span>
                                                    <span className="text-emerald-400 text-sm font-bold">
                                                        {(t.confidence * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex-1 flex gap-4">
                                            <div className="flex-1">
                                                <p className="text-xs text-center mb-2 text-slate-400 uppercase tracking-wide">Cropped Focus</p>
                                                <img src={t.cropped_b64} className="w-full h-32 object-contain rounded-lg bg-black border border-slate-700/50" alt="Crop" />
                                            </div>
                                            <div className="flex-1">
                                                <p className="text-xs text-center mb-2 text-slate-400 uppercase tracking-wide">Grad-CAM Heatmap</p>
                                                <img src={t.gradcam_b64} className="w-full h-32 object-contain rounded-lg bg-black border border-slate-700/50" alt="GradCAM" />
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <div className="h-full border-2 border-dashed border-slate-700/50 rounded-xl flex flex-col items-center justify-center text-slate-500 bg-slate-800/20 min-h-[500px]">
                            <svg className="w-20 h-20 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                            <p className="text-lg">Awaiting MRI Scan</p>
                            <p className="text-sm mt-2 max-w-sm text-center">Results for detection, segmentation, and explainability will appear here after analysis.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
