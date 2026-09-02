import { useNavigate } from "react-router-dom";


function PropertyCard({
    icon,
    title,
    description,
    path,
    full
}) {


    const navigate = useNavigate();


    return (

        <div

            className={full ? "card card-full" : "card"}

            onClick={() => navigate(path)}

        >

            <span className="icon">

                {icon}

            </span>


            <h3>

                {title}

            </h3>


            <p>

                {description}

            </p>


        </div>

    );

}


export default PropertyCard;