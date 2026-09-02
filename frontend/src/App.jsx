import {
    Routes,
    Route
} from "react-router-dom";


import Home from "./pages/Home";

import Density from "./pages/Density";

import Heat from "./pages/Heat";

import Surface from "./pages/Surface";

import Solubility from "./pages/Solubility";



function App(){


    return (

        <Routes>


            {/* 首页 */}

            <Route

                path="/"

                element={<Home />}

            />



            {/* 密度 */}

            <Route

                path="/density"

                element={<Density />}

            />



            {/* 后续页面 */}

            <Route

                path="/heat"

                element={<Heat />}

            />



            <Route

                path="/surface"

                element={<Surface />}

            />



            <Route

                path="/solubility"

                element={<Solubility />}

            />



            <Route

                path="/other"

                element={

                    <div style={{padding:"50px"}}>

                        Other Models Page

                    </div>

                }

            />


        </Routes>

    );


}


export default App;