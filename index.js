const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const port = 5002;

// Setup function to initialize proxy
const setupProxy = (app) => {
    app.use('/api', createProxyMiddleware({
        target: 'http://localhost:5001',
        changeOrigin: true,
        logLevel: 'debug',
        onError: (err, req, res) => {
            console.error('Error in proxy middleware', err);
            res.status(500).send('Proxy encountered an error');
        }
    }));
};

app.set('view engine', 'ejs');
app.use(express.static('public'));
app.use('/css', express.static(__dirname + 'node_modules/bootstrap/dist/css'));

app.get('/', (req, res) => {
    res.render('index');
});

setupProxy(app); // Applying the proxy setup

app.listen(port, () => console.log(`Server running on port ${port}.`));