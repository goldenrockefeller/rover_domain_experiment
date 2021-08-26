#ifndef GOLDENROCKEFELLER_POLICYOPT_FLAT_NETWORK_HPP 
#define GOLDENROCKEFELLER_POLICYOPT_FLAT_NETWORK_HPP

#include <valarray>
#include <vector>
#include <memory>


namespace goldenrockefeller {
	namespace policyopt {
		struct Experience {
			std::valarray<double> observation;
			std::valarray<double> action;
			double reward;
		};

		struct FlatNetwork {

			std::vector<std::valarray<double>> linear;
			std::valarray<double> fixed_layer;
			std::valarray<double> bias0;
			double bias1;
			double leaky_scale;

			FlatNetwork();
			FlatNetwork(std::size_t n_in_dims, std::size_t n_hidden_units);

			std::unique_ptr<FlatNetwork> copy() const;

			std::size_t n_in_dims() const;
			std::size_t n_hidden_units() const;

			std::size_t n_parameters() const;
			std::valarray<double> parameters() const;
			void set_parameters(const std::valarray<double>& parameters);

			double eval(const std::valarray<double>& input) const;

			std::valarray<double> grad_wrt_parameters(
				const std::valarray<double>& input,	
				double output_grad
			);
		};

		struct Approximator {
		protected:
			virtual Approximator* copy_impl() const = 0;
		public:
			virtual ~Approximator() {};

			std::unique_ptr<Approximator> copy() const{
				return std::unique_ptr<Approximator>(this->copy_impl());
			}

			virtual std::valarray<double> parameters() const = 0;
			virtual void set_parameters(const std::valarray<double>& parameters) = 0;
			virtual std::size_t n_parameters() const = 0;

			virtual double eval(const Experience& input) const = 0;
			virtual std::valarray<double> grad_wrt_parameters(
				const Experience& input,
				double output_grad
			) = 0;

			virtual void update(const std::vector<Experience>& experiences) = 0;
			virtual void update(const Experience& experience, double target_value);
			
		};

		struct FlatNetworkOptimizer {
			double time_horizon;
			double epsilon;
			double learning_rate;
			std::valarray<double> pressures;
			int learning_mode; // 0 - unconditioned, 1 - conditioned, 2 - conditioned & accelerated

			FlatNetworkOptimizer(const std::valarray<double>& init_parameters);

			void add_pressures(const std::valarray<double>& grad);
			void add_pressures(const std::vector<std::valarray<double>>& grads);
			std::valarray<double> delta_parameters(const std::valarray<double>& grad, double error) const;
			void discount_pressures();
		};

		struct FlatNetworkApproximator: public Approximator {
		protected:
			virtual FlatNetworkApproximator* copy_impl() const override;
		public:			

			FlatNetworkOptimizer optimizer;
			std::shared_ptr<FlatNetwork> flat_network;


			FlatNetworkApproximator();
			FlatNetworkApproximator(std::shared_ptr<FlatNetwork> flat_network);
			FlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units);
			virtual ~FlatNetworkApproximator() {};

			std::unique_ptr<FlatNetworkApproximator> copy() const {
				return std::unique_ptr<FlatNetworkApproximator>(this->copy_impl());
			}

			virtual std::valarray<double> parameters() const override;
			virtual void set_parameters(const std::valarray<double>& parameters) override;
			virtual std::size_t n_parameters() const override;

			virtual double eval(const Experience& input) const override;
			virtual std::valarray<double> grad_wrt_parameters(
				const Experience& input,
				double output_grad
			) override;

			virtual void update(const std::vector<Experience>& experiences) override;
			virtual void update(const Experience& experience, double target_value) override;

		};

		struct MonteFlatNetworkApproximator : public FlatNetworkApproximator {
		protected:
			virtual MonteFlatNetworkApproximator* copy_impl() const override;
		public:
			MonteFlatNetworkApproximator() : FlatNetworkApproximator() {}

			MonteFlatNetworkApproximator(std::shared_ptr<FlatNetwork> flat_network) 
				: FlatNetworkApproximator(flat_network)
				{}

			MonteFlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units)
				: FlatNetworkApproximator(n_in_dims, n_hidden_units)
				{}

			virtual ~MonteFlatNetworkApproximator() {};


			std::unique_ptr<MonteFlatNetworkApproximator> copy() const {
				return std::unique_ptr<MonteFlatNetworkApproximator>(this->copy_impl());
			}

			virtual void update(const std::vector<Experience>& experiences) override;
		};


		struct DiscountFlatNetworkApproximator : public FlatNetworkApproximator {
		protected:
			virtual DiscountFlatNetworkApproximator* copy_impl() const override;
		public:
			double discount_factor = 0.97;

			DiscountFlatNetworkApproximator() : FlatNetworkApproximator() {}

			DiscountFlatNetworkApproximator(std::shared_ptr<FlatNetwork> flat_network) 
				: FlatNetworkApproximator(flat_network)
				{}

			DiscountFlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units)
				: FlatNetworkApproximator(n_in_dims, n_hidden_units)
				{}

			virtual ~DiscountFlatNetworkApproximator() {};


			std::unique_ptr<DiscountFlatNetworkApproximator> copy() const {
				return std::unique_ptr<DiscountFlatNetworkApproximator>(this->copy_impl());
			}

			virtual void update(const std::vector<Experience>& experiences) override;
		};


		struct QFlatNetworkApproximator : public FlatNetworkApproximator {
		protected:
			virtual QFlatNetworkApproximator* copy_impl() const override;
		public:

			QFlatNetworkApproximator() : FlatNetworkApproximator() {}

			QFlatNetworkApproximator(std::shared_ptr<FlatNetwork> flat_network)
				: FlatNetworkApproximator(flat_network)
			{}

			QFlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units)
				: FlatNetworkApproximator(n_in_dims, n_hidden_units)
			{}

			virtual ~QFlatNetworkApproximator() {};

			std::unique_ptr<QFlatNetworkApproximator> copy() const {
				return std::unique_ptr<QFlatNetworkApproximator>(this->copy_impl());
			}


			virtual void update(const std::vector<Experience>& experiences) override;
		};

		struct UFlatNetworkApproximator : public FlatNetworkApproximator {
		protected:
			virtual UFlatNetworkApproximator* copy_impl() const override;
		public:
			UFlatNetworkApproximator() : FlatNetworkApproximator() {}

			UFlatNetworkApproximator(std::shared_ptr<FlatNetwork> flat_network)
				: FlatNetworkApproximator(flat_network)
			{}

			UFlatNetworkApproximator(std::size_t n_in_dims, std::size_t n_hidden_units)
				: FlatNetworkApproximator(n_in_dims, n_hidden_units)
			{}

			virtual ~UFlatNetworkApproximator() {};


			std::unique_ptr<UFlatNetworkApproximator> copy() const {
				return std::unique_ptr<UFlatNetworkApproximator>(this->copy_impl());
			}

			virtual double eval(const Experience& input) const override;
			virtual std::valarray<double> grad_wrt_parameters(
				const Experience& input,
				double output_grad
			) override;


			virtual void update(const std::vector<Experience>& experiences) override;
		};

		struct UqFlatNetworkApproximator : public Approximator {
		protected:
			virtual UqFlatNetworkApproximator* copy_impl() const override;
		public:
			std::shared_ptr<UFlatNetworkApproximator> u_approximator;
			std::shared_ptr<QFlatNetworkApproximator> q_approximator;

			UqFlatNetworkApproximator();
			UqFlatNetworkApproximator(
				std::shared_ptr<UFlatNetworkApproximator> u_approximator, 
				std::shared_ptr<QFlatNetworkApproximator> q_approximator
			);
			virtual ~UqFlatNetworkApproximator() {};


			std::unique_ptr<UqFlatNetworkApproximator> copy() const {
				return std::unique_ptr<UqFlatNetworkApproximator>(this->copy_impl());
			}

			virtual std::valarray<double> parameters() const override;
			virtual void set_parameters(const std::valarray<double>& parameters) override;
			virtual std::size_t n_parameters() const override;

			virtual double eval(const Experience& input) const override;
			virtual std::valarray<double> grad_wrt_parameters(
				const Experience& input,
				double output_grad
			) override;

			virtual void update(const std::vector<Experience>& experiences) override;
		};
	}
}

#endif