import Banner from "./Banner/Banner";

import CoinsTable from "./Coinstable";

export default function Homepage() {

  return (
    <div>
      
      <Banner/>
      <br /><br />
      <CoinsTable/>
     
      {/*<ParallaxImages images={imageUrls} baseVelocity={-5} />*/}
      

    </div>
    
  );
}
