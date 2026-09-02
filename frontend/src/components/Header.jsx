import { Link } from "react-router-dom";


function Header({ title }) {


    return (

        <div className="header-bar">


            <div className="container d-flex align-items-center">


                <Link

                    to="/"

                    className="system-title"

                >

                    {title}

                </Link>



                <span className="ms-auto badge bg-secondary version-badge">

                    Version 3.0.0 | Skywalker

                </span>


            </div>


        </div>

    );


}


export default Header;