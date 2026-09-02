import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import App from './App'

import "bootstrap/dist/css/bootstrap.min.css";

import "./styles/global.css";


ReactDOM.createRoot(
    document.getElementById('root')
).render(

    <BrowserRouter>

        <App />

    </BrowserRouter>

)