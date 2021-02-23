#include "relu_network_approximator.hpp"
#include <valarray>
#include <vector>
#include <sstream>
#include <exception>
#include <random>


using std::valarray;
using std::vector;
using std::size_t;
using std::unique_ptr;
using std::shared_ptr;
using std::slice;
using std::move;
using std::exp;
using std::ostringstream;
using std::invalid_argument;
using std::runtime_error;
using std::make_shared;


namespace rockefeg {
	namespace policyopt {
		ReluNetwork::ReluNetwork() : ReluNetwork::ReluNetwork(1, 1, 1) {}

		ReluNetwork::ReluNetwork(size_t n_in_dims, size_t n_hidden_units, size_t n_out_dims) : leaky_scale(0.01) {
			if (n_in_dims <= 0) {
				ostringstream msg;
				msg << "The number of input dimensions (n_in_dims = "
					<< n_in_dims
					<< ") must be positive.";
				throw invalid_argument(msg.str());
			}

			if (n_hidden_units <= 0) {
				ostringstream msg;
				msg << "The number of hidden units (n_hidden_units = "
					<< n_hidden_units
					<< ") must be positive.";
				throw invalid_argument(msg.str());
			}

			if (n_out_dims <= 0) {
				ostringstream msg;
				msg << "The number of output dimensions (n_out_dims = "
					<< n_out_dims
					<< ") must be positive.";
				throw invalid_argument(msg.str());
			}

			this->linear0.resize(n_hidden_units);

			for (valarray<double>& weights : this->linear0) {
				weights.resize(n_in_dims);
			}

			this->linear1.resize(n_out_dims);

			for (valarray<double>& weights : this->linear1) {
				weights.resize(n_hidden_units);
			}

			this->bias0.resize(n_hidden_units);
			this->bias1.resize(n_out_dims);


			std::random_device rd;  //Will be used to obtain a seed for the random number engine
			std::mt19937_64 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<> distrib(-1., 1.);

			// Initialize linear weights.
			for (valarray<double>& weights : this->linear0) {
				for (double& weight : weights) {
					weight = distrib(gen);
				}
			}

			for (valarray<double>& weights : this->linear1) {
				for (double& weight : weights) {
					weight = distrib(gen);
				}
			}

			for (double& val : this->bias0) {
				val = distrib(gen);
			}

			for (double& val : this->bias1) {
				val = distrib(gen);
			}
		}

		unique_ptr<ReluNetwork> ReluNetwork::copy() const {
			unique_ptr<ReluNetwork> new_relu_network_core{
				new ReluNetwork(this->n_in_dims(), this->n_hidden_units(), this->n_out_dims())
			};

			new_relu_network_core->set_parameters(this->parameters());

			return move(new_relu_network_core);
		}

		size_t ReluNetwork::n_in_dims() const {
			return this->linear0[0].size();
		}

		size_t ReluNetwork::n_hidden_units() const {
			return this->linear0.size();
		}

		size_t ReluNetwork::n_out_dims() const {
			return this->linear1.size();
		}

		size_t ReluNetwork::n_parameters() const {
			size_t n_parameters{ 0 };

			n_parameters += this->linear0[0].size() * this->linear0.size();
			n_parameters += this->linear1[0].size() * this->linear1.size();
			n_parameters += this->bias0.size();
			n_parameters += this->bias1.size();

			return n_parameters;
		}

		valarray<double> ReluNetwork::parameters() const {
			// Order linear0, bias0, linear1, bias1
			valarray<double> parameters;
			size_t slice_start{ 0 };
			size_t slice_length{ 0 };

			parameters.resize(this->n_parameters());

			for (const valarray<double>& weights : this->linear0) {
				slice_length = weights.size();
				parameters[slice(slice_start, slice_length, 1)] = weights;
				slice_start += slice_length;
			}

			slice_length = this->bias0.size();
			parameters[slice(slice_start, slice_length, 1)] = this->bias0;
			slice_start += slice_length;

			for (const valarray<double>& weights : this->linear1) {
				slice_length = weights.size();
				parameters[slice(slice_start, slice_length, 1)] = weights;
				slice_start += slice_length;
			}

			slice_length = this->bias1.size();
			parameters[slice(slice_start, slice_length, 1)] = this->bias1;
			slice_start += slice_length;

			if (slice_start != this->n_parameters()) {
				ostringstream msg;
				msg << "Something went wrong. slice_start (slice_start = "
					<< slice_start
					<< ") should be equal to the number of parameters for the neural network "
					<< "(this->n_parameters() =  "
					<< this->n_parameters()
					<< ").";
				throw runtime_error(msg.str());
			}

			return parameters;
		}


		void ReluNetwork::set_parameters(const valarray<double>& parameters) {
			size_t slice_start{ 0 };
			size_t slice_length{ 0 };

			if (parameters.size() != this->n_parameters()) {
				ostringstream msg;
				msg << "The number of setting parameters (parameters.size() = "
					<< parameters.size()
					<< ") must be equal to the number of parameters for the neural network "
					<< "(this->n_parameters() =  "
					<< this->n_parameters()
					<< ").";
				throw invalid_argument(msg.str());
			}

			for (valarray<double>& weights : this->linear0) {
				slice_length = weights.size();
				weights = parameters[slice(slice_start, slice_length, 1)];
				slice_start += slice_length;
			}

			slice_length = this->bias0.size();
			this->bias0 = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;

			for (valarray<double>& weights : this->linear1) {
				slice_length = weights.size();
				weights = parameters[slice(slice_start, slice_length, 1)];
				slice_start += slice_length;
			}

			slice_length = this->bias1.size();
			this->bias1 = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;


			if (slice_start != this->n_parameters()) {
				ostringstream msg;
				msg << "Something went wrong. slice_start (slice_start = "
					<< slice_start
					<< ") should be equal to the number of parameters for the neural network "
					<< "(this->n_parameters() =  "
					<< this->n_parameters()
					<< ").";
				throw runtime_error(msg.str());
			}

		}

		
		valarray<double> ReluNetwork::eval(const valarray<double>& input) const {
			size_t n_hidden_units{ this->n_hidden_units() };
			size_t n_out_dims{ this->n_out_dims() };
			size_t n_in_dims{ this->n_in_dims() };
			


			if (input.size() != n_in_dims) {
				ostringstream msg;
				msg << "The size of the input (input.size() = "
					<< input.size()
					<< ") must be equal to the number of input dimensions for the neural network "
					<< "(this->n_in_dims() =  "
					<< n_in_dims
					<< ").";
				throw invalid_argument(msg.str());
			}

			// Res0 is the output of the first layer (linear0, bia0, relu).
			valarray<double> res0;
			res0.resize(n_hidden_units);

			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] = (this->linear0[i] * input).sum();
			}

			res0 += this->bias0;

			// Preform Relu. 
			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] = res0[i] * (res0[i] > 0.) + res0[i] * (res0[i] <= 0.) * this->leaky_scale;
			}

			// Res1 is the output of the second layer (linear1, bias1) given Res0 (output of first layer).
			valarray<double> res1;
			res1.resize(n_out_dims);

			for (size_t i{ 0 }; i < n_out_dims; i++) {
				res1[i] = (this->linear1[i] * res0).sum();
			}

			res1 += this->bias1;

			return res1;
		}

		std::valarray<double> ReluNetwork::grad_wrt_parameters(
			const std::valarray<double>& input,
			const std::valarray<double>& output_grad
		) {
			// Order linear0, bias0, linear1, bias1
			size_t n_hidden_units{ this->n_hidden_units() };
			size_t n_out_dims{ this->n_out_dims() };
			size_t n_in_dims{ this->n_in_dims() };
			

			if (input.size() != n_in_dims) {
				ostringstream msg;
				msg << "The size of the input (input.size() = "
					<< input.size()
					<< ") must be equal to the number of input dimensions for the neural network "
					<< "(this->n_in_dims() =  "
					<< n_in_dims
					<< ").";
				throw invalid_argument(msg.str());
			}

			if (output_grad.size() != n_out_dims) {
				ostringstream msg;
				msg << "The size of the output gradient (output_grad.size() = "
					<< output_grad.size()
					<< ") must be equal to the number of output dimensions for the neural network "
					<< "(this->n_out_dims() =  "
					<< n_out_dims
					<< ").";
				throw invalid_argument(msg.str());
			}
		
			// Res0 is the output of the first layer (linear0, bia0, relu).
			valarray<double> res0;
			res0.resize(n_hidden_units);

			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] = (this->linear0[i] * input).sum();
			}

			res0 += this->bias0;

			// Pre Relu Result
			valarray<double> pre_relu_res{res0};

			// Preform Relu. 
			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] = res0[i] * (res0[i] > 0.) + res0[i] * (res0[i] <= 0.) * this->leaky_scale;
			}

			// Get intermediate gradients.
			valarray<double> grad_wrt_res0;
			grad_wrt_res0.resize(n_hidden_units);
			grad_wrt_res0 = 0.;

			for (size_t i{ 0 }; i < n_out_dims; i++) {
				grad_wrt_res0 += this->linear1[i] * output_grad[i];
			}

			valarray<double> grad_wrt_pre_relu_res;
			grad_wrt_pre_relu_res.resize(n_hidden_units);
			
			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				grad_wrt_pre_relu_res[i] 
					+= grad_wrt_res0[i] * (pre_relu_res[i] > 0.)
					+ grad_wrt_res0[i] * (pre_relu_res[i] <= 0.) * this->leaky_scale;
			}


			// Get parameter gradients
			size_t n_parameters{ this->n_parameters() };
			size_t slice_start{ 0 };
			size_t slice_length{ 0 };
			valarray<double> grad;
			grad.resize(n_parameters);

			// Gradient for linear0
			for (size_t i{0}; i < n_hidden_units; i++) {
				slice_length = n_in_dims;
				grad[slice(slice_start, slice_length, 1)] = grad_wrt_pre_relu_res[i] * input;
				slice_start += slice_length;
			}

			// Gradient for Bias0
			slice_length = n_hidden_units;
			grad[slice(slice_start, slice_length, 1)] = grad_wrt_pre_relu_res;
			slice_start += slice_length;

			// Gradient for linear1
			for (size_t i{ 0 }; i < n_out_dims; i++) {
				slice_length = n_hidden_units;
				grad[slice(slice_start, slice_length, 1)] = output_grad[i] * res0;
				slice_start += slice_length;
			}

			// Gradient for Bias1
			slice_length = n_out_dims;
			grad[slice(slice_start, slice_length, 1)] = output_grad;
			slice_start += slice_length;

			if (slice_start != grad.size()) {
				ostringstream msg;
				msg << "Something went wrong. slice_start (slice_start = "
					<< slice_start
					<< ") should be equal to the number of parameters for "
					<< "the neural network "
					<< "(this->n_parameters() =  "
					<< n_parameters
					<< ").";
				throw runtime_error(msg.str());
			}

			return grad;
		}

		ReluNetworkApproximator::ReluNetworkApproximator() :
			ReluNetworkApproximator::ReluNetworkApproximator(
				make_shared<ReluNetwork>()
			)
		{}

		ReluNetworkApproximator::ReluNetworkApproximator(
			shared_ptr<ReluNetwork> relu_network
		) :

			relu_network(relu_network),
			learning_rate(1.e-5)
		{}

		ReluNetworkApproximator::ReluNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units, std::size_t n_out_dims) :
			ReluNetworkApproximator::ReluNetworkApproximator(
				make_shared<ReluNetwork>(n_in_dims, n_hidden_units, n_out_dims)
			)
		{}

		unique_ptr<ReluNetworkApproximator> ReluNetworkApproximator::copy() const {
			return unique_ptr<ReluNetworkApproximator> {
				new ReluNetworkApproximator(move(this->relu_network->copy()))
			};
		}

		valarray<double> ReluNetworkApproximator::parameters() const {
			return this->relu_network->parameters();
		}


		void ReluNetworkApproximator::set_parameters(const valarray<double>& parameters) {
			this->relu_network->set_parameters(parameters);
		}

		size_t ReluNetworkApproximator::n_parameters() const {
			return this->relu_network->n_parameters();
		}

		valarray<double> ReluNetworkApproximator::eval(const valarray<double>& input) const {
			return this->relu_network->eval(input);
		}

		void ReluNetworkApproximator::update(const valarray<double>& observation_action, double feedback) {
			valarray<double> grad = this->relu_network->grad_wrt_parameters(observation_action, valarray<double>(feedback, 1));
			valarray<double> parameters = this->relu_network->parameters();
			this->relu_network->set_parameters(parameters - this->learning_rate * grad);
		}
		



	} // namespace policyopt
} // namespace rockefeg