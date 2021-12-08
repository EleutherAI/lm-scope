import React from 'react';
import {
    BrowserRouter,
    Routes,
    Route,
  } from "react-router-dom";
import Home from './Home';
  import Layout from "./Layout";



function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route
                    path={`${process.env.PUBLIC_URL}/viewer`}
                    element={<Layout>
                        <div>viewer</div>
                    </Layout>}
                />
                <Route
                    path={`${process.env.PUBLIC_URL}/`}
                    element={<Layout>
                        <Home />
                    </Layout>}
                />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
