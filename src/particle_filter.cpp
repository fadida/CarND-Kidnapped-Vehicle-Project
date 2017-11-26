/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

//#define ENABLE_DEBUG_SET_ASSOCIATIONS
//#define ENABLE_DEBUG_PRINTS

// standard deviation array indices.
constexpr int STD_ARR_X_IDX     = 0;
constexpr int STD_ARR_Y_IDX     = 1;
constexpr int STD_ARR_THETA_IDX = 2;

constexpr int INVALID_IDX = -1;

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//  Set the number of particles. Initialize all particles to first position (based on estimates of
	//  x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  normal_distribution<double> noisy_x(x, std[STD_ARR_X_IDX]);
  normal_distribution<double> noisy_y(y, std[STD_ARR_Y_IDX]);
  normal_distribution<double> noisy_theta(theta, std[STD_ARR_THETA_IDX]);

	num_particles = 10;

	cout << "Initializing particle filter with " << num_particles << " particles." << endl;

	for (int particle_idx = 0; particle_idx < num_particles; ++particle_idx) {

	  Particle particle;

	  particle.id     = particle_idx + 1;
	  particle.x      = noisy_x(gen);
	  particle.y      = noisy_y(gen);
	  particle.theta  = noisy_theta(gen);
	  particle.weight = 1.0;

	  particles.push_back(particle);
	  weights.push_back(particle.weight);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  normal_distribution<double> noisy_x(0, std_pos[STD_ARR_X_IDX]);
  normal_distribution<double> noisy_y(0, std_pos[STD_ARR_Y_IDX]);
  normal_distribution<double> noisy_theta(0, std_pos[STD_ARR_THETA_IDX]);

  double factor;     /* factor for location equations - dependent on yaw_rate. */
  double delta_yaw;  /* The change in yaw over delta_t */

  if (yaw_rate) {
    factor    = velocity / yaw_rate;
    delta_yaw = yaw_rate * delta_t;
  } else {
    factor = velocity * delta_t;
  }

#ifdef ENABLE_DEBUG_PRINTS
  cout << "moving particles with vel=" << velocity << " yaw_rate=" << yaw_rate << endl;
#endif

  for (int particle_idx = 0; particle_idx < particles.size(); ++particle_idx) {

    // Add noise to location and orientation.
    particles[particle_idx].x     += noisy_x(gen);
    particles[particle_idx].y     += noisy_y(gen);
    particles[particle_idx].theta += noisy_theta(gen);

    if (yaw_rate) {
      particles[particle_idx].x += factor * (sin(particles[particle_idx].theta + delta_yaw) - sin(particles[particle_idx].theta));
      particles[particle_idx].y += factor * (cos(particles[particle_idx].theta) - cos(particles[particle_idx].theta + delta_yaw));

      particles[particle_idx].theta += delta_yaw;
    } else {
      particles[particle_idx].x += factor * cos(particles[particle_idx].theta);
      particles[particle_idx].y += factor * sin(particles[particle_idx].theta);
    }

  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// For each observation, find the closest landmark from `predicted` vector.

  for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
    LandmarkObs& obs      = observations.at(obs_idx);
    double       min_dist = INFINITY;

    for (int pred_idx = 0; pred_idx < predicted.size(); ++pred_idx) {
      LandmarkObs& pred_obs  = predicted[pred_idx];
      double       pred_dist = dist(pred_obs.x, pred_obs.y, obs.x, obs.y);

      if (pred_dist < min_dist) {
        min_dist = pred_dist;
        obs.id = pred_obs.id;
      }

    }

  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // Update the weights of each particle using a multi-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html


  // Various factors to help speed up the weights calculation.
  double lmark_std_x            = std_landmark[STD_ARR_X_IDX];
  double lmark_std_y            = std_landmark[STD_ARR_Y_IDX];
  double lmark_std2_x           = 2.0 * lmark_std_x * lmark_std_x;
  double lmark_std2_y           = 2.0 * lmark_std_y * lmark_std_y;
  double gaussian_normalization = 1.0 / (2.0 * M_PI * lmark_std_x * lmark_std_y);

#ifdef ENABLE_DEBUG_PRINTS
  static int tick = 0;            /* tick counter. */
  int        predicted_match = 0; /* Number of particles that that sense the same number of
                                     landmarks as the observations vector. */
#endif

  for (int particle_idx = 0; particle_idx < num_particles; ++particle_idx) {

    Particle& particle = particles[particle_idx];

    // Collect the landmark this particle observes.
    vector<LandmarkObs> observedLandmarks;
    for (int lmark_idx = 0; lmark_idx < map_landmarks.landmark_list.size(); ++lmark_idx) {
      const Map::single_landmark_s& lmark = map_landmarks.landmark_list.at(lmark_idx);

      double lmark_dist = dist(particle.x, particle.y, lmark.x_f, lmark.y_f);
      if (lmark_dist <= sensor_range) {
        LandmarkObs lmarkObs;
        lmarkObs.id = lmark_idx;
        lmarkObs.x  = lmark.x_f;
        lmarkObs.y  = lmark.y_f;

        observedLandmarks.push_back(lmarkObs);
      }
    }

    // Check if the particle observed the same number of landmarks as the car.
    if (observations.size() == observedLandmarks.size()) {

#ifdef ENABLE_DEBUG_PRINTS
      ++predicted_match;
#endif

      double cos_theta = cos(particle.theta);
      double sin_theta = sin(particle.theta);

      // Calculate observations from particle POV.
      vector<LandmarkObs> particleObservations(observations.size());
      for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
        const LandmarkObs& obs = observations.at(obs_idx);

        LandmarkObs particleObs;
        particleObs.id = INVALID_IDX;
        particleObs.x  = particle.x + (cos_theta * obs.x) - (sin_theta * obs.y);
        particleObs.y  = particle.y + (sin_theta * obs.x) + (cos_theta * obs.y);

        particleObservations[obs_idx] = particleObs;
      }

      // Associate landmarks to observations.
      dataAssociation(observedLandmarks, particleObservations);

#ifdef ENABLE_DEBUG_SET_ASSOCIATIONS
      vector<int> particle_associations(observedLandmarks.size());
      vector<double> particle_sense_x(observedLandmarks.size());
      vector<double> particle_sense_y(observedLandmarks.size());
#endif

      // Calculate particle weight
      particle.weight = 1;
      for (int obs_idx = 0; obs_idx < particleObservations.size(); ++obs_idx) {
        const LandmarkObs& obs = particleObservations.at(obs_idx);

        const Map::single_landmark_s& lmark = map_landmarks.landmark_list.at(obs.id);

        double delta_x = obs.x - lmark.x_f;
        double delta_y = obs.y - lmark.y_f;

        double exp_x = (delta_x * delta_x) / lmark_std2_x;
        double exp_y = (delta_y * delta_y) / lmark_std2_y;

        double obs_weight = gaussian_normalization * exp(-exp_x - exp_y);

        particle.weight *= obs_weight;

#ifdef ENABLE_DEBUG_SET_ASSOCIATIONS
        particle_associations[obs_idx] = lmark.id_i;
        particle_sense_x[obs_idx]      = (obs.x);
        particle_sense_y[obs_idx]      = (obs.y);
#endif

      }

#ifdef ENABLE_DEBUG_SET_ASSOCIATIONS
      SetAssociations(particle, particle_associations, particle_sense_x, particle_sense_x);
#endif

    } else {
      // We reached here if number of particle observations differ for the
      // car number of observations, which means that this particle location is
      // no good.
      // We set its weight to zero in order to lower the probability that this
      //particle will be sampled during the re-sample stage..
      particle.weight = 0;
    }

    weights[particle_idx] = particle.weight;
  }


#ifdef ENABLE_DEBUG_PRINTS
  cout << "Update weights(): tick=" << tick << " sum="
      << weights_sum << " pred_match=" << predicted_match <<
      " num_particles=" << num_particles << " num_obs="<<observations.size() << endl;
  ++tick;
#endif

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine      gen;
  discrete_distribution<int> weight_dist(weights.begin(), weights.end());
  vector<Particle>           sampled_particles(num_particles);
  int                        sampling_count = num_particles;

#ifdef ENABLE_DEBUG_PRINTS
  cout << "resampling particles" << endl;
#endif

  for(int particle_idx = 0; particle_idx < num_particles; ++particle_idx)
  {
    int sampled_idx = weight_dist(gen);
    sampled_particles[particle_idx] = particles[sampled_idx];
    sampled_particles[particle_idx].id = particle_idx + 1;
  }

  // Replace the filter particles with the sampled particles.
  particles = sampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
