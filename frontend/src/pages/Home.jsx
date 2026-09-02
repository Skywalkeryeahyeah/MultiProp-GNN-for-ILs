import "../styles/home.css";
import PropertyCard from "../components/PropertyCard";


function Home() {

    return (
        <>

            <div className="top-stripe"></div>

            <header>
                <h1>GNN Prediction Hub</h1>

                <p className="sub-title">
                    Advanced Machine Learning Framework for Ionic Liquids
                </p>
            </header>


            <div className="main-container">

                <div className="grid">


                    <PropertyCard
                        icon="⚖"
                        title="MASS DENSITY"
                        description="Predictive modeling of volumetric mass distribution (ρ)"
                        path="/density"
                    />


                    <PropertyCard
                        icon="⚗"
                        title="HEAT CAPACITY"
                        description={
                            <>
                                Molar isobaric heat capacity determination (C<sub>p</sub>)
                            </>
                        }
                        path="/heat"
                    />


                    <PropertyCard
                        icon="☍"
                        title="SURFACE TENSION"
                        description="Interfacial force and energy prediction (σ)"
                        path="/surface"
                    />


                    <PropertyCard
                        icon="⚙"
                        title="OTHER MODELS"
                        description="Additional thermodynamic & transport properties"
                        path="/other"
                    />

                </div>



                <div className="section-divider">

                    <span>
                        CORE RESEARCH
                    </span>

                </div>



                <div className="grid">

                    <PropertyCard
                        full={true}
                        icon="⚛"
                        title="GAS SOLUBILITY"
                        description="Equilibrium mole fraction of gas absorption (x) - Advanced GNN Estimation"
                        path="/solubility"
                    />

                </div>


            </div>



            <footer>

                © 2026 Integrated GNN Framework (MultiProp-GNN) for Ionic Liquids Research

            </footer>


        </>
    );
}


export default Home;