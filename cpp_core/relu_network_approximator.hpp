#ifndef ROCKEFEG_POLICYOPT_RELU_NETWORK_HPP 
#define ROCKEFEG_POLICYOPT_RELU_NETWORK_HPP

#include <valarray>
#include <vector>
#include <memory>


namespace rockefeg {
	namespace policyopt {
		struct ReluNetwork {

			std::vector<std::valarray<double>> linear0;
			std::valarray<double> bias0;
			std::vector<std::valarray<double>> linear1;
			std::valarray<double> bias1;
			double leaky_scale;

			ReluNetwork();
			ReluNetwork(std::size_t n_in_dims, std::size_t n_hidden_units, std::size_t n_out_dims);

			std::unique_ptr<ReluNetwork> copy() const;

			std::size_t n_in_dims() const;
			std::size_t n_hidden_units() const;
			std::size_t n_out_dims() const;
			std::size_t n_parameters() const;
			std::valarray<double> parameters() const;
			void set_parameters(const std::valarray<double>& parameters);

			std::valarray<double> eval(const std::valarray<double>& input) const;

			std::valarray<double> grad_wrt_parameters(
				const std::valarray<double>& input, 
				const std::valarray<double>& output_grad
			);
		};

		struct ReluNetworkApproximator {
			std::shared_ptr<ReluNetwork> relu_network;

			double learning_rate;

			ReluNetworkApproximator();
			ReluNetworkApproximator(std::shared_ptr<ReluNetwork> relu_network);
			ReluNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units, std::size_t n_out_dims);

			std::unique_ptr<ReluNetworkApproximator> copy() const;

			std::valarray<double> parameters() const;
			void set_parameters(const std::valarray<double>& parameters);
			std::size_t n_parameters() const;

			std::valarray<double> eval(const std::valarray<double>& input) const;
			void update(const std::valarray<double>& observation_action, double feedback);
		};
	}
}

#endif