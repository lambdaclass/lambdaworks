// Initialize Mermaid diagrams
(function() {
    // Load Mermaid from CDN
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
    script.onload = function() {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
        // Re-run mermaid on page load
        mermaid.run();
    };
    document.head.appendChild(script);
})();
