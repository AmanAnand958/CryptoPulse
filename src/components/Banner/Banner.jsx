import React, { useRef } from 'react';
import Carousel from './Carousel';
import './Banner.css'; // Import the CSS file for animations

function Banner() {
  const scrollRef = useRef(null); // Create a ref for the scroll target

  const handleScrollDown = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' }); // Smooth scroll to the target section
    }
  };

  return (
    <div className="banner-container">
      <div className="relative w-full h-[calc(100vh-100px)] overflow-hidden">
        <video
          className="absolute inset-0 object-cover w-full h-full -z-10"
          autoPlay
          loop
          muted
          preload="auto"
          playsInline
          onError={(e) => console.error("Video failed to load", e)}
        >
          <source src="/background.mp4" type="video/mp4"/>
          Your browser does not support the video tag.
        </video>
        <div className="absolute inset-0 bg-gray-900 bg-opacity-50"></div>
        <div className="relative flex flex-col items-center justify-center h-full text-center text-white">
          <h1 className="animated-text">
            <span>always be</span>
            <div className="message">
              <div className="word1">Informed</div>
              <div className="word2">Monitoring</div>
              <div className="word3">Tracking</div>
            </div>
          </h1>
          <p className="text-lg text-gray-300 capitalize font-montserrat">
            Navigate the Waves of Cryptocurrency with Focused Insights
          </p>
          <br/><br/><br/><br/><br/><br/><br/>
          <button className="scroll" onClick={handleScrollDown}>
          </button>
        </div>
      </div>
      <div ref={scrollRef} className="scroll-target">
        <Carousel />
      </div>
    </div>
  );
}

export default Banner;
