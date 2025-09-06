import React from 'react';
import './index.css';

export const FullScreenLoader = () => {
  return (
    <div className="loader-wrapper fullscreen">
      <div className="gradient-orb"></div>
    </div>
  );
};

export const ButtonLoader = () => {
  return <div className="gradient-orb small"></div>;
};
