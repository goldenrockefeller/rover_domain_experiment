#include "flat_network_approximator.hpp"
#include <valarray>
#include <vector>
#include <sstream>
#include <exception>
#include <random>


using std::valarray;
using std::vector;
using std::size_t;
using std::pow;
using std::unique_ptr;
using std::shared_ptr;
using std::slice;
using std::move;
using std::exp;
using std::ostringstream;
using std::invalid_argument;
using std::runtime_error;
using std::make_shared;
using std::sqrt;


namespace goldenrockefeller {
	namespace policyopt {
		valarray<double> concatenate(const valarray<double>& a, const valarray<double>& b) {
			valarray<double> c(0., a.size() + b.size());
			
			c[slice(0, a.size(), 1)] = a;
			c[slice(a.size(), b.size(), 1)] = b;

			return c;
		}

		FlatNetwork::FlatNetwork() : FlatNetwork::FlatNetwork(1, 1) {}

		FlatNetwork::FlatNetwork(size_t n_in_dims, size_t n_hidden_units) : leaky_scale(0.1) {
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

			this->linear.resize(n_hidden_units);

			for (valarray<double>& weights : this->linear) {
				weights.resize(n_in_dims);
			}

			this->bias0.resize(n_hidden_units);
			this->fixed_layer.resize(n_hidden_units);

			std::random_device rd;  //Will be used to obtain a seed for the random number engine
			std::mt19937_64 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<> in_distrib(-pow(3. / n_in_dims,0.5), pow(3. / n_in_dims,0.5));
			std::uniform_real_distribution<> hidden_distrib(-pow(3. / n_hidden_units,0.5), pow(3. / n_hidden_units,0.5));

			// Initialize linear weights.
			for (valarray<double>& weights : this->linear) {
				for (double& weight : weights) {
					weight = in_distrib(gen)  ;
				}
			}

			for (double& val : this->bias0) {
				val = in_distrib(gen);				
			}

			this->bias1 = hidden_distrib(gen);

			// Fixed layer. Half of the fixed weights are 1, other half is 
			size_t half_fixed_layer_size = fixed_layer.size() / 2; // integer division
			this->fixed_layer = 1.;
			for (size_t id = 0; id < half_fixed_layer_size; id ++) {
				this->fixed_layer[id] = -1;
			}
		}

		unique_ptr<FlatNetwork> FlatNetwork::copy() const {
			unique_ptr<FlatNetwork> new_flat_network_core{
				new FlatNetwork(this->n_in_dims(), this->n_hidden_units())
			};

			new_flat_network_core->set_parameters(this->parameters());
			new_flat_network_core->leaky_scale = this->leaky_scale;

			return move(new_flat_network_core);
		}

		size_t FlatNetwork::n_in_dims() const {
			return this->linear[0].size();
		}

		size_t FlatNetwork::n_hidden_units() const {
			return this->linear.size();
		}

		size_t FlatNetwork::n_parameters() const {
			size_t n_parameters{ 0 };

			n_parameters += this->linear[0].size() * this->linear.size();
			n_parameters += this->bias0.size();
			n_parameters += 1; // for this->bias1

			return n_parameters;
		}

		valarray<double> FlatNetwork::parameters() const {
			// Order linear, bias0, bias1
			valarray<double> parameters;
			size_t slice_start{ 0 };
			size_t slice_length{ 0 };

			parameters.resize(this->n_parameters());

			for (const valarray<double>& weights : this->linear) {
				slice_length = weights.size();
				parameters[slice(slice_start, slice_length, 1)] = weights;
				slice_start += slice_length;
			}

			slice_length = this->bias0.size();
			parameters[slice(slice_start, slice_length, 1)] = this->bias0;
			slice_start += slice_length;

			parameters[slice_start] = this->bias1;
			slice_start += 1;

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


		void FlatNetwork::set_parameters(const valarray<double>& parameters) {
			// Order linear, bias0, bias1
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

			for (valarray<double>& weights : this->linear) {
				slice_length = weights.size();
				weights = parameters[slice(slice_start, slice_length, 1)];
				slice_start += slice_length;
			}

			slice_length = this->bias0.size();
			this->bias0 = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;

			this->bias1 = parameters[slice_start];
			slice_start += 1;


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


		double FlatNetwork::eval(const valarray<double>& input) const {
			size_t n_hidden_units{ this->n_hidden_units() };
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

			// Res0 is the output of the first layer (linear, bia0, relu).
			valarray<double> res0;
			res0.resize(n_hidden_units);

			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] =  (this->linear[i] * input).sum();
			}

			res0 += this->bias0;

			// Perform Relu. 
			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				res0[i] = res0[i] * (res0[i] > 0.) + res0[i] * (res0[i] <= 0.) * this->leaky_scale;
			}

			// Res1 is the output of the second layer (bias1) given Res0 (output of first layer).
			double res1;
			res1 = (res0 * this->fixed_layer).sum() + this->bias1;

			return res1;
		}

		std::valarray<double> FlatNetwork::grad_wrt_parameters(
			const std::valarray<double>& input,
			double output_grad
		) {
			// Order linear, bias0, bias1
			size_t n_hidden_units{ this->n_hidden_units() };\
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
				res0[i] = (this->linear[i] * input).sum();
			}

			res0 += this->bias0;

			// Pre Relu Result
			valarray<double> pre_relu_res{ res0 };
			
			// Get intermediate gradients.
			valarray<double> grad_wrt_post_relu_res;
			grad_wrt_post_relu_res.resize(n_hidden_units);
			grad_wrt_post_relu_res = output_grad * this->fixed_layer;

			valarray<double> grad_wrt_pre_relu_res;
			grad_wrt_pre_relu_res.resize(n_hidden_units);

			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				grad_wrt_pre_relu_res[i]
					+= grad_wrt_post_relu_res[i] * (pre_relu_res[i] > 0.)
					+ grad_wrt_post_relu_res[i] * (pre_relu_res[i] <= 0.) * this->leaky_scale;
			}


			// Get parameter gradients
			size_t n_parameters{ this->n_parameters() };
			size_t slice_start{ 0 };
			size_t slice_length{ 0 };
			valarray<double> grad;
			grad.resize(n_parameters);

			// Gradient for linear
			for (size_t i{ 0 }; i < n_hidden_units; i++) {
				slice_length = n_in_dims;
				grad[slice(slice_start, slice_length, 1)] = grad_wrt_pre_relu_res[i] * input;
				slice_start += slice_length;
			}

			// Gradient for Bias0
			slice_length = n_hidden_units;
			grad[slice(slice_start, slice_length, 1)] = grad_wrt_pre_relu_res;
			slice_start += slice_length;

			// Gradient for Bias1
			grad[slice_start] = output_grad;
			slice_start += 1;

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

		void Approximator::update(const Experience& experience, double target_value) {
			throw runtime_error("not implemented");
		}

		FlatNetworkOptimizer::FlatNetworkOptimizer(const valarray<double>& init_parameters) :
			time_horizon(100),
			epsilon(1e-9),
			learning_rate(0.5),
			pressures(0., init_parameters.size()),
			learning_mode(0)
		{}
		
		void FlatNetworkOptimizer::add_pressures(const valarray<double>& grad) {
			if (this->learning_mode == 1 || this->learning_mode == 2) {
				// (1) conditioned or (2) conditioned & accelerated
				valarray<double> local_pressures = grad * grad;
				valarray<double> relative_pressures = local_pressures / (this->pressures + local_pressures + this->epsilon);

				double sum = relative_pressures.sum();
				double sum_of_sqr = (relative_pressures * relative_pressures).sum();

				double inv_step_size = sum * sum / (sum_of_sqr + this->epsilon);


				this->pressures += local_pressures * inv_step_size;
			}
		}

		void FlatNetworkOptimizer::add_pressures(const vector<valarray<double>>& grads) {
			if (this->learning_mode == 1 || this->learning_mode == 2) {
				// (1) conditioned or (2) conditioned & accelerated
				valarray<double> basic_pressures = this->pressures;

				for (const valarray<double>& grad : grads) {
					basic_pressures += grad * grad;
				}

				for (const valarray<double>& grad : grads) {
					valarray<double> local_pressures = grad * grad;
					valarray<double> relative_pressures = local_pressures / (basic_pressures + this->epsilon);

					double sum = relative_pressures.sum();
					double sum_of_sqr = (relative_pressures * relative_pressures).sum();

					double inv_step_size = sum * sum / (sum_of_sqr + this->epsilon);

					this->pressures += local_pressures * inv_step_size;
				}
			}
		}

		valarray<double> FlatNetworkOptimizer::delta_parameters(const valarray<double>& grad, double error) const {
			// Must add pressures before hand.

			valarray<double> delta_parameters;
			if (this->learning_mode == 0) {
				// (0) unconditioned
				delta_parameters = grad * error * this->learning_rate;

			}
			else if (this->learning_mode == 1 || this->learning_mode == 2) {
				// (1) conditioned or (2) conditioned & accelerated
				delta_parameters = this->learning_rate * grad * error  / (this->pressures + this->epsilon);
				
			}

			return delta_parameters;
		}

		void FlatNetworkOptimizer::discount_pressures() {
			if (this->learning_mode == 1 || this->learning_mode == 2) {
				// (1) conditioned or (2) conditioned & accelerated
				this->pressures *= 1. - 1. / this->time_horizon;
			}
		}


		FlatNetworkApproximator::FlatNetworkApproximator() :
			FlatNetworkApproximator::FlatNetworkApproximator(
				make_shared<FlatNetwork>()
			)
		{}

		FlatNetworkApproximator::FlatNetworkApproximator(
			shared_ptr<FlatNetwork> flat_network
		) :

			flat_network(flat_network),
			optimizer(flat_network->parameters())
		{}

		FlatNetworkApproximator::FlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units) :
			FlatNetworkApproximator::FlatNetworkApproximator(
				make_shared<FlatNetwork>(n_in_dims, n_hidden_units)
			)
		{}

		FlatNetworkApproximator* FlatNetworkApproximator::copy_impl() const {
			return new FlatNetworkApproximator(move(this->flat_network->copy()));
		}

		valarray<double> FlatNetworkApproximator::parameters() const {
			return this->flat_network->parameters();
		}


		void FlatNetworkApproximator::set_parameters(const valarray<double>& parameters) {
			this->flat_network->set_parameters(parameters);
		}

		size_t FlatNetworkApproximator::n_parameters() const {
			return this->flat_network->n_parameters();
		}

		double FlatNetworkApproximator::eval(const Experience& input) const {
			return this->flat_network->eval(concatenate(input.observation, input.action));
		}

		valarray<double> FlatNetworkApproximator::grad_wrt_parameters(const Experience& input, double output_grad) {
			return this->flat_network->grad_wrt_parameters(concatenate(input.observation, input.action), output_grad);
		}

		void FlatNetworkApproximator::update(const std::vector<Experience>& experiences) {
			double sample_fitness = 0.;
			double traj_eval = 0.;
			size_t n_steps = experiences.size();

			for (const Experience& experience : experiences) {
				sample_fitness += experience.reward;
				traj_eval += this->eval(experience);
			}
			traj_eval /= n_steps;

			double error = sample_fitness - traj_eval;

			valarray<double> grad(0., this->n_parameters());

			for (const Experience& experience : experiences) {
				grad += this->grad_wrt_parameters(experience, 1.);
			}

			this->optimizer.add_pressures(grad);
			valarray<double> delta_parameters = this->optimizer.delta_parameters(grad, error);
			this->optimizer.discount_pressures();

			valarray<double> parameters = this->parameters();
			this->set_parameters(parameters + delta_parameters);
			
		}

		void FlatNetworkApproximator::update(const Experience& experience, double target_value) {
			double eval = this->eval(experience);

			double error = target_value - eval;

			valarray<double> grad = this->grad_wrt_parameters(experience, error);
			valarray<double> delta_parameters = grad * this->optimizer.learning_rate;

			// this->optimizer.add_pressures(grad);
			// valarray<double> delta_parameters = this->optimizer.delta_parameters(grad, error);
			// this->optimizer.discount_pressures();

			this->set_parameters(this->parameters() + delta_parameters);
		}

		MonteFlatNetworkApproximator* MonteFlatNetworkApproximator::copy_impl() const {
			return new MonteFlatNetworkApproximator(move(this->flat_network->copy()));
		}

		void MonteFlatNetworkApproximator::update(const std::vector<Experience>& experiences) {
			double sample_fitness = 0.;
			double traj_eval = 0.;
			valarray<double> delta_parameters(0., this->n_parameters());

			vector<valarray<double>> grads;
			grads.reserve(experiences.size());

			for (const Experience& experience : experiences) {
				sample_fitness += experience.reward;
			
				grads.push_back(this->grad_wrt_parameters(experience, 1.));
			}
			this->optimizer.add_pressures(grads);

			size_t experience_id = 0;
			for (const Experience& experience : experiences) {

				double traj_eval = this->eval(experience);
				double error = sample_fitness - traj_eval;
				valarray<double> grad = grads[experience_id];
				delta_parameters += this->optimizer.delta_parameters(grad, error);
				experience_id += 1;
			}
			this->optimizer.discount_pressures();

			valarray<double> parameters = this->parameters() + delta_parameters;
			this->set_parameters(parameters);
		}

		DiscountFlatNetworkApproximator* DiscountFlatNetworkApproximator::copy_impl() const {
			return new DiscountFlatNetworkApproximator(move(this->flat_network->copy()));
		}

		void DiscountFlatNetworkApproximator::update(const std::vector<Experience>& experiences) {
			double sample_fitness = 0.;
			double step_eval = 0.;
			valarray<double> delta_parameters(0., this->n_parameters());

			vector<valarray<double>> grads;
			grads.reserve(experiences.size());

			for (const Experience& experience : experiences) {
				

				grads.push_back(this->grad_wrt_parameters(experience, 1.));
			}
			this->optimizer.add_pressures(grads);

			size_t experience_id = 0;
			for (const Experience& experience : experiences) {
				sample_fitness = 0.;
				size_t other_experience_id = 0;
				for (const Experience& other_experience : experiences) {
					other_experience_id += 1;
					if (other_experience_id >= experience_id) {
						sample_fitness += other_experience.reward * pow(this->discount_factor, other_experience_id- experience_id);
					}
				}
				double step_eval = this->eval(experience);
				double error = sample_fitness - step_eval;
				valarray<double> grad = grads[experience_id];
				delta_parameters += this->optimizer.delta_parameters(grad, error);
				experience_id += 1;
			}
			this->optimizer.discount_pressures();


			valarray<double> parameters = this->parameters();
			this->set_parameters(parameters + delta_parameters);
		}



		QFlatNetworkApproximator* QFlatNetworkApproximator::copy_impl() const {
			return	new QFlatNetworkApproximator(move(this->flat_network->copy()));
		}

		void QFlatNetworkApproximator::update(const std::vector<Experience>& experiences) {
			size_t n_steps = experiences.size();
			valarray<double> delta_parameters(0., this->n_parameters());

			vector<valarray<double>> grads;
			grads.reserve(experiences.size());

			for (const Experience& experience : experiences) {
				valarray<double> grad = this->grad_wrt_parameters(experience, 1.);
				
				grads.push_back(this->grad_wrt_parameters(experience, 1.));
			}
			this->optimizer.add_pressures(grads);

			size_t experience_id = 0;
			for (size_t step_id = 0; step_id < n_steps; step_id ++) {
				const  Experience& experience = experiences[step_id];
				double td_error;

				if (step_id == n_steps - 1) {
					td_error = experience.reward - this->eval(experience);
				} else {
					const  Experience& next_experience = experiences[step_id + 1];

					td_error = experience.reward + this->eval(next_experience) - this->eval(experience);
				}
				valarray<double> grad = grads[experience_id];
				delta_parameters += this->optimizer.delta_parameters(grad, td_error);
				experience_id += 1;
			}
			this->optimizer.discount_pressures();

			valarray<double> parameters = this->parameters();
			this->set_parameters(parameters + delta_parameters);
		}


		UFlatNetworkApproximator* UFlatNetworkApproximator::copy_impl() const {
			return new UFlatNetworkApproximator(move(this->flat_network->copy()));
		}

		double UFlatNetworkApproximator::eval(const Experience& input) const {
			return this->flat_network->eval(input.observation);
		}

		valarray<double> UFlatNetworkApproximator::grad_wrt_parameters(const Experience& input, double output_grad) {
			return this->flat_network->grad_wrt_parameters(input.observation, output_grad);
		}

		void UFlatNetworkApproximator::update(const std::vector<Experience>& experiences) {
			size_t n_steps = experiences.size();
			valarray<double> delta_parameters(0., this->n_parameters());

			vector<valarray<double>> grads;
			grads.reserve(experiences.size());

			for (const Experience& experience : experiences) {
				valarray<double> grad = this->grad_wrt_parameters(experience, 1.);
				grads.push_back(this->grad_wrt_parameters(experience, 1.));
			}
			this->optimizer.add_pressures(grads);
			
			size_t experience_id = 0;
			for (size_t step_id = 0; step_id < n_steps; step_id++) {
				const  Experience& experience = experiences[step_id];
				double td_error;

				if (step_id == 0) {
					td_error = - this->eval(experience);
				}
				else {
					const  Experience& prev_experience = experiences[step_id - 1];

					td_error = prev_experience.reward + this->eval(prev_experience) - this->eval(experience);
				}

				valarray<double> grad = grads[experience_id];
				delta_parameters += this->optimizer.delta_parameters(grad, td_error);
				experience_id += 1;
			}
			this->optimizer.discount_pressures();

			valarray<double> parameters = this->parameters();
			this->set_parameters(parameters + delta_parameters);
		}

		UqFlatNetworkApproximator::UqFlatNetworkApproximator() :
			UqFlatNetworkApproximator(
				make_shared<UFlatNetworkApproximator>(), make_shared<QFlatNetworkApproximator>()
			)
		{}

		UqFlatNetworkApproximator::UqFlatNetworkApproximator(
			shared_ptr<UFlatNetworkApproximator> u_approximator,
			shared_ptr<QFlatNetworkApproximator> q_approximator
		) : u_approximator(u_approximator),
			q_approximator(q_approximator)
		{}

		UqFlatNetworkApproximator* UqFlatNetworkApproximator::copy_impl() const {
			return (
				new UqFlatNetworkApproximator(
					move(this->u_approximator->copy()),
					move(this->q_approximator->copy())
				)
			);
		}

		valarray<double> UqFlatNetworkApproximator::parameters() const {
			return concatenate(this->u_approximator->parameters(), this->q_approximator->parameters());
		}

		void UqFlatNetworkApproximator::set_parameters(const valarray<double>& parameters) {
			if (parameters.size() != this->n_parameters()) {
				ostringstream msg;
				msg << "The number of setting parameters (parameters.size() = "
					<< parameters.size()
					<< ") must be equal to the number of parameters for the approximator "
					<< "(this->n_parameters() =  "
					<< this->n_parameters()
					<< ").";
				throw invalid_argument(msg.str());
			}

			this->u_approximator->set_parameters(parameters[slice(0, this->u_approximator->n_parameters(), 1)]);
			
			this->q_approximator->set_parameters(
				parameters[
					slice(
						this->u_approximator->n_parameters(),
						this->q_approximator->n_parameters(), 
						1
					)
				]
			);

		}

		size_t UqFlatNetworkApproximator::n_parameters() const {
			return this->u_approximator->n_parameters() + this->q_approximator->n_parameters();
		}

		double UqFlatNetworkApproximator::eval(const Experience& input) const {
			return this->u_approximator->eval(input) + this->q_approximator->eval(input);
		}

		valarray<double> UqFlatNetworkApproximator::grad_wrt_parameters(
			const Experience& input,
			double output_grad
		) {
			return (
				this->u_approximator->grad_wrt_parameters(input, output_grad)
				+ this->q_approximator->grad_wrt_parameters(input, output_grad)
			);
		}

		void UqFlatNetworkApproximator::update(const vector<Experience>& experiences) {
			this->u_approximator->update(experiences);
			this->q_approximator->update(experiences);
		}

	} // namespace policyopt
} // namespace goldenrockefeller