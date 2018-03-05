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

#define EPS 0.00001
#define NUMBER_OF_PARTICLES 100
#define INITIAL_WEIGHT 1.0

using namespace std;

default_random_engine gen(time(0));

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (initialized()) {
    return;
  }

	num_particles = NUMBER_OF_PARTICLES;

	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// Creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for(unsigned i = 0; i < num_particles; ++i){
		Particle particle;
		particle.id = i;

		// Sample from these normal distributions
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		particle.weight = INITIAL_WEIGHT;

		particles.push_back(particle);
	}

	weights.resize(num_particles, INITIAL_WEIGHT);

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// Creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(0.0, std_x);
	normal_distribution<double> dist_y(0.0, std_y);
	normal_distribution<double> dist_theta(0.0, std_theta);

	for(unsigned i = 0; i < num_particles; ++i){
		Particle particle = particles[i];
		double x = particle.x;
		double y = particle.y;
		double theta = particle.theta;

		// Predict next state
		if (fabs(yaw_rate) < EPS) { // yaw is not changing / Moving straight
			particle.x += velocity * delta_t * cos(theta);
			particle.y += velocity * delta_t * sin(theta);
    } else {
			particle.x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particle.y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
		}
		particle.theta += yaw_rate * delta_t;

		// Add noise
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

		particles[i] = particle;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned i = 0; i < observations.size(); ++i) {
		double min_dist = numeric_limits<double>::max();
		int mapId = -1;
		for (unsigned j = 0; j < predicted.size(); ++j) {
			double x_y_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (x_y_dist < min_dist) {
				min_dist = x_y_dist;
				mapId = predicted[j].id;
			}
		}
		observations[i].id = mapId;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
	for(unsigned i = 0; i < num_particles; ++i){
		Particle particle = particles[i];
		double p_x = particle.x;
		double p_y = particle.y;
		double p_t = particle.theta;

		// Consider landmarks only if in sensor range from the particle
    vector<LandmarkObs> predicted_landmarks;
    for(Map::single_landmark_s lm : map_landmarks.landmark_list){
			double lm_dist = dist(p_x, p_y, lm.x_f, lm.y_f);
      if (lm_dist < sensor_range) {
        predicted_landmarks.push_back((LandmarkObs){.id = lm.id_i, .x = lm.x_f, .y = lm.y_f});
      }
    }

		// Transform the observations in the world/map coordinates
		vector<LandmarkObs> trans_observed_landmarks;
		for(LandmarkObs observation : observations){
				double o_x = observation.x;
				double o_y = observation.y;
				LandmarkObs trans_observation;
				trans_observation.x = p_x + o_x * cos(p_t) - o_y * sin(p_t);
				trans_observation.y = p_y + o_x * sin(p_t) + o_y * cos(p_t);
				trans_observed_landmarks.push_back(trans_observation);
		}

		// Associate the observed and predicted landmarks
		dataAssociation(predicted_landmarks, trans_observed_landmarks);

		// call SetAssociations
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		for(unsigned i = 0; i < trans_observed_landmarks.size(); ++i){
			associations.push_back(trans_observed_landmarks[i].id);
			sense_x.push_back(trans_observed_landmarks[i].x);
			sense_y.push_back(trans_observed_landmarks[i].y);
		}
		particle = SetAssociations(particle, associations, sense_x, sense_y);

		// Calculate the weights
		double std_l_range = std_landmark[0];
  	double std_l_bearing = std_landmark[1];
    double gauss_norm= 1 / (2 * M_PI * std_l_range * std_l_bearing);
		double double_std_r_2 = 2 * std_l_range * std_l_range;
		double double_std_b_2 = 2 * std_l_bearing * std_l_bearing;
		particle.weight = INITIAL_WEIGHT;
		for(unsigned i = 0; i < trans_observed_landmarks.size(); ++i){
				LandmarkObs obs = trans_observed_landmarks[i];
				LandmarkObs pred;
				// find predicted landmark corresponding to the observed landmark
				for (int j = 0; j < predicted_landmarks.size(); ++j){
					if (obs.id == predicted_landmarks[j].id){
						pred = predicted_landmarks[j];
						break;
					}
				}

				double dx = obs.x - pred.x;
				double dy = obs.y - pred.y;

				double exponent = dx * dx / double_std_r_2 + dy * dy / double_std_b_2;
				double weight = gauss_norm * exp(-exponent);

	      particle.weight *= weight;
		}
		particles[i] = particle;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	weights.clear();
	for(unsigned i = 0; i < num_particles; ++i){
		Particle particle = particles[i];
		weights.push_back(particle.weight);
	}

	discrete_distribution<> weight_distribution(weights.begin(), weights.end()); // No need to normalize, done automatically in discrete_distribution

	std::vector<Particle> resampled_particles;
	for (unsigned i = 0; i < num_particles; ++i) {
		int index = weight_distribution(gen);
		resampled_particles.push_back(particles[index]);
	}
	particles = resampled_particles; // TODO: std::move ?
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

		return particle;
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
