import { useState } from "react";
import axios from "axios";

import Header from "../components/Header";
import ResultPanel from "../components/ResultPanel";

import "../styles/prediction.css";


function Density(){


    const [cation,setCation] = useState("");

    const [anion,setAnion] = useState("");

    const [temperature,setTemperature] = useState(298.15);

    const [pressure,setPressure] = useState(0.1);

    const [result,setResult] = useState("--");

    const [loading,setLoading] = useState(false);



    async function runInference(){


        setLoading(true);


        const payload={

            type:"density",

            cation_smiles:cation,

            anion_smiles:anion,

            T:Number(temperature),

            P:Number(pressure),

            X:1.0

        };



        try{


            const response = await axios.post(

                "http://127.0.0.1:5000/predict",

                payload

            );


            const data=response.data;


            if(data.status==="success"){

                setResult(data.value);

            }

            else{

                alert(data.message);

            }


        }

        catch(error){

            console.log(error);

            alert(
                "Backend Error. Check Flask server."
            );

        }


        finally{

            setLoading(false);

        }


    }




    return (

        <>


        <Header

            title="Ionic Liquid Density Prediction System"

        />



        <div className="container">


            <div className="row g-5">


                {/* 左侧输入 */}

                <div className="col-md-6">


                    <div className="academic-card">


                        <div className="section-title">

                            Experimental Configuration

                        </div>



                        <div className="mb-4">


                            <label className="fw-bold">

                                Cation SMILES Notation

                            </label>


                            <input

                                type="text"

                                className="form-control"

                                placeholder="e.g. CCCCn1cc[n+](C)c1"

                                value={cation}

                                onChange={
                                    e=>setCation(e.target.value)
                                }

                            />


                        </div>




                        <div className="mb-4">


                            <label className="fw-bold">

                                Anion SMILES Notation

                            </label>


                            <input

                                type="text"

                                className="form-control"

                                placeholder="e.g. F[P-](F)(F)(F)(F)F"

                                value={anion}

                                onChange={
                                    e=>setAnion(e.target.value)
                                }

                            />


                        </div>





                        <div className="mb-4">


                            <label className="fw-bold">

                                Isothermal Temperature (T) / K

                            </label>



                            <div className="param-row">


                                <input

                                    type="range"

                                    className="form-range flex-grow-1"

                                    min="273.15"

                                    max="473.15"

                                    step="0.01"

                                    value={temperature}

                                    onChange={
                                        e=>setTemperature(e.target.value)
                                    }

                                />


                                <input

                                    type="number"

                                    className="input-precise"

                                    value={temperature}

                                    step="0.01"

                                    onChange={
                                        e=>setTemperature(e.target.value)
                                    }

                                />


                            </div>


                        </div>






                        <div className="mb-5">


                            <label className="fw-bold">

                                Isobaric Pressure (P) / MPa

                            </label>



                            <div className="param-row">


                                <input

                                    type="range"

                                    className="form-range flex-grow-1"

                                    min="0.1"

                                    max="100"

                                    step="0.1"

                                    value={pressure}

                                    onChange={
                                        e=>setPressure(e.target.value)
                                    }

                                />



                                <input

                                    type="number"

                                    className="input-precise"

                                    value={pressure}

                                    step="0.1"

                                    onChange={
                                        e=>setPressure(e.target.value)
                                    }

                                />


                            </div>


                        </div>






                        <button

                            className="btn-predict w-100"

                            onClick={runInference}

                            disabled={loading}


                        >

                            {

                                loading

                                ?

                                "CALCULATING..."

                                :

                                "RUN GNN INFERENCE"

                            }


                        </button>



                    </div>


                </div>







                {/* 右侧结果 */}


                <div className="col-md-6">


                    <ResultPanel

                        title="Predicted Mass Density (ρ)"

                        value={result}

                        unit="kg / m³"

                        loading={loading}

                    />


                </div>



            </div>


        </div>



        </>

    )


}


export default Density;