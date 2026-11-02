// Small static reference, deliberately not a substitute for Julia type inference.
export const builtins: Record<string, { signature: string; parameters: string[]; doc: string }> = {
  println: { signature: 'println([io::IO], xs...)', parameters: ['io::IO', 'xs...'], doc: 'Print values followed by a newline.' },
  print: { signature: 'print([io::IO], xs...)', parameters: ['io::IO', 'xs...'], doc: 'Write a text representation of values.' },
  length: { signature: 'length(collection)', parameters: ['collection'], doc: 'Return the number of elements in a collection.' },
  size: { signature: 'size(A, [dim])', parameters: ['A', 'dim'], doc: 'Return the dimensions of an array, or the size along one dimension.' },
  zeros: { signature: 'zeros([T=Float64], dims...)', parameters: ['T', 'dims...'], doc: 'Create an array filled with zeros.' },
  ones: { signature: 'ones([T=Float64], dims...)', parameters: ['T', 'dims...'], doc: 'Create an array filled with ones.' },
  map: { signature: 'map(f, collections...)', parameters: ['f', 'collections...'], doc: 'Apply a function to each element of the collections.' },
  filter: { signature: 'filter(f, collection)', parameters: ['f', 'collection'], doc: 'Keep elements for which the predicate returns true.' },
  sum: { signature: 'sum([f,] collection)', parameters: ['f', 'collection'], doc: 'Sum elements, optionally after applying a function.' },
  'push!': { signature: 'push!(collection, items...)', parameters: ['collection', 'items...'], doc: 'Append items to a collection.' },
  'pop!': { signature: 'pop!(collection)', parameters: ['collection'], doc: 'Remove and return the last item.' },
  include: { signature: 'include(path::AbstractString)', parameters: ['path::AbstractString'], doc: 'Evaluate a Julia source file in the current module. Workbench indexes literal include paths without evaluating them.' },
  typeof: { signature: 'typeof(x)', parameters: ['x'], doc: 'Return the concrete type of a value.' },
  eltype: { signature: 'eltype(collection)', parameters: ['collection'], doc: 'Return the element type of a collection.' },
  collect: { signature: 'collect(collection)', parameters: ['collection'], doc: 'Collect elements into an array.' },
  eachindex: { signature: 'eachindex(A...)', parameters: ['A...'], doc: 'Create an efficient iterator for indexing the arrays.' },
  enumerate: { signature: 'enumerate(iter)', parameters: ['iter'], doc: 'Iterate over pairs of index and value.' },
  zip: { signature: 'zip(iters...)', parameters: ['iters...'], doc: 'Iterate over several iterators together.' },
  error: { signature: 'error(message...)', parameters: ['message...'], doc: 'Raise an error with the supplied message.' },
  sqrt: { signature: 'sqrt(x)', parameters: ['x'], doc: 'Return the square root.' },
  abs: { signature: 'abs(x)', parameters: ['x'], doc: 'Return the absolute value.' }
};
export const types = ['Int', 'Int8', 'Int16', 'Int32', 'Int64', 'UInt', 'UInt8', 'Float16', 'Float32', 'Float64', 'Bool', 'Char', 'String', 'Symbol', 'Nothing', 'Missing', 'Any', 'Number', 'Real', 'Integer', 'AbstractString', 'AbstractArray', 'Vector', 'Matrix', 'Array', 'Tuple', 'NamedTuple', 'Dict', 'Set', 'IO'];
