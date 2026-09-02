import { useState } from "react";
import axios from "axios";

import Header from "../components/Header";
import ResultPanel from "../components/ResultPanel";

import "../styles/prediction.css";



function Solubility(){


    const [cation,setCation] = useState("");

    const [anion,setAnion] = useState("");

    const [temperature,setTemperature] = useState(313.15);

    const [pressure,setPressure] = useState(1.0);

    const [result,setResult] = useState("--");

    const [loading,setLoading] = useState(false);





    async function runInference(){


        setLoading(true);



        const payload = {


            type:"solubility",


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



            const data = response.data;



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

            title="Gas Solubility Prediction System (CO₂)"

        />





        <div className="container">


            <div className="row g-5">





                {/* 输入区域 */}



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

                                placeholder="e.g. CCCCCCn1cc[n+](C)c1"

                                value={cation}

                                onChange={
                                    e=>setCation(
                                        e.target.value
                                    )
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

                                placeholder="e.g. C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)(F)"

                                value={anion}

                                onChange={
                                    e=>setAnion(
                                        e.target.value
                                    )
                                }

                            />


                        </div>








                        <div className="mb-4">


                            <label className="fw-bold">

                                Temperature (T) / K

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
                                        e=>setTemperature(
                                            e.target.value
                                        )
                                    }

                                />




                                <input

                                    type="number"

                                    className="input-precise"

                                    step="0.01"

                                    value={temperature}

                                    onChange={
                                        e=>setTemperature(
                                            e.target.value
                                        )
                                    }

                                />


                            </div>


                        </div>









                        <div className="mb-5">


                            <label className="fw-bold">

                                Pressure (P) / MPa

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
                                        e=>setPressure(
                                            e.target.value
                                        )
                                    }

                                />




                                <input

                                    type="number"

                                    className="input-precise"

                                    step="0.1"

                                    value={pressure}

                                    onChange={
                                        e=>setPressure(
                                            e.target.value
                                        )
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









                {/* 结果区域 */}



                <div className="col-md-6">


                    <ResultPanel


                        title="Gas Solubility (mole fraction)"


                        value={result}


                        unit="mol gas / mol IL"


                        loading={loading}


                    />


                </div>





            </div>


        </div>



        </>


    );



}



export default Solubility;