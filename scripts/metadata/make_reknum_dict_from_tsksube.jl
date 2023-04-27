using Distributed

filename = ""#define here filename

function getcol()
    columns = 0
    open(filename) do stream
        for outer columns in eachline(stream)
            break
        end
    end
    return columns
end

columns = split(getcol(), "\t")
println("column : $columns")

function extract_icu_recnum(filename, columns)
    mydict = Dict()
    recnum_idx = findall( x -> x == "RECNUM", columns)
    proccode_idx = findall( x -> x == "PROCCODE", columns)
    open(filename) do stream
        for (i, ln) in enumerate(eachline(stream))
            if i==1
                continue
            elseif i>1000000
                break 
            else
                # vectstr = Dict(zip(columns, split(ln, "\t")))
                vectstr = split(ln, "\t")
                opr_codes = get(mydict, vectstr[recnum_idx], Set())
                mydict[vectstr[recnum_idx]] = push!(opr_codes, vectstr[proccode_idx])
            end 
        end
    end
    
    recnums = []
    for k in keys(mydict)
        if "NABE" in mydict[k] || "NABE" in  mydict[k]
            push!(recnums, k)
        end
    end
    return recnums
end

@time extract_icu_recnum(filename, columns)
@time extract_icu_recnum(filename, columns)
