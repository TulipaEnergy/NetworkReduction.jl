@testset "NL case results tests" begin
    case_name = "NL_case"
    input_dir = joinpath(INPUT_FOLDER, case_name)
    output_dir = joinpath(OUTPUT_FOLDER, case_name)
    reference_dir = joinpath(output_dir, "reference_outputs")

    main_full_analysis(input_dir, output_dir)

    @testset "Equivalent Capacities Match" begin
        generated_path = joinpath(output_dir, "Equivalent_Capacities_QP.csv")
        reference_path = joinpath(reference_dir, "Equivalent_Capacities_QP.csv")

        @test isfile(generated_path)
        @test isfile(reference_path)

        generated = CSV.read(generated_path, DataFrame)
        reference = CSV.read(reference_path, DataFrame)

        sort!(generated, [:synth_line_from, :synth_line_to])
        sort!(reference, [:synth_line_from, :synth_line_to])

        @test size(generated) == size(reference)
        @test generated[:, [:synth_line_from, :synth_line_to]] ==
              reference[:, [:synth_line_from, :synth_line_to]]

        @test all(isapprox.(generated.C_eq_pu, reference.C_eq_pu; atol = 1e-6, rtol = 1e-6))
    end

    @testset "TTC Comparison Match" begin
        generated_path = joinpath(output_dir, "TTC_Comparison_QP.csv")
        reference_path = joinpath(reference_dir, "TTC_Comparison_QP.csv")

        @test isfile(generated_path)
        @test isfile(reference_path)

        generated = CSV.read(generated_path, DataFrame)
        reference = CSV.read(reference_path, DataFrame)

        sort!(generated, [:From_Name, :To_Name])
        sort!(reference, [:From_Name, :To_Name])

        @test size(generated) == size(reference)
        @test generated[:, [:From_Name, :To_Name]] == reference[:, [:From_Name, :To_Name]]

        @test all(
            isapprox.(
                generated.TTC_Original_pu,
                reference.TTC_Original_pu;
                atol = 1e-6,
                rtol = 1e-6,
            ),
        )

        @test all(
            isapprox.(
                generated.TTC_Equivalent_pu,
                reference.TTC_Equivalent_pu;
                atol = 1e-6,
                rtol = 1e-6,
            ),
        )
    end
end
