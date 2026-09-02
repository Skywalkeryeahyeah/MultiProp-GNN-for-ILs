function ResultPanel({

    title,

    value,

    unit,

    loading

}) {


    return (

        <div className="academic-card result-panel">


            <div className="result-label">

                {title}

            </div>



            {

                loading &&

                <div className="loading">

                    Calculating...

                </div>

            }



            <div className="result-container">


                <div className="result-value">

                    {value}

                </div>



                <div className="unit">

                    {unit}

                </div>


            </div>



            <div className="stats-footer">


                <div>

                    GNN Architecture: MultiProp-GNN

                </div>


                <div>

                    Confidence Interval: ±0.5% | Source: IL-GNN-v2

                </div>


            </div>



        </div>

    );


}


export default ResultPanel;