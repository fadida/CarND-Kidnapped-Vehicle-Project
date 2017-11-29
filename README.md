# Kidnapped Vehicle Project
Self-Driving Car Engineer Nanodegree Program

This project works with Udacity self driving simulator (for term 2) and
uses a particle filter in order to localize a vehicle.

## The algorithm
The first step is to initialize a particle filter with 20 particles using GPS coordinates with uniform weights.
After the filter is initialized, the particles weights are calculated based on the car observations and the map and then 
resampled based on the weights.
When the car move, the filter prediction function is called with the car velocity and yaw rate.
The particle filter is using the bicycle model in order to model the car movement.



